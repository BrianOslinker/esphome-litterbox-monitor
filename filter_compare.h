#pragma once
// Local only, temporary: the box tracker's stillness test run on five
// versions of the same readings, to judge which filter earns its place:
// unfiltered, median of 3 (what the tracker uses), median of 5, mean of 3
// and mean of 5.

#include <algorithm>
#include <cstdio>
#include "state_analyzer.h"

class FilterCompare {
 public:
  static constexpr int N = 5;
  enum { RAW, MED3, MED5, AVG3, AVG5 };

  /** Every reading, unfiltered. `limit_g` is the Still Limit in use (med3). */
  void process(float raw_g, float limit_g) {
    for (int i = 4; i > 0; i--) last_[i] = last_[i - 1];
    last_[0] = raw_g;
    if (have_ < 5) have_++;
    float out[N] = {raw_g, median(3), median(5), mean(3), mean(5)};
    bool probing = probe_left_ > 0;
    if (probing) {
      float dev = fabsf(raw_g - out[MED3]);
      if (dev > SPIKE_G) spikes_++;
      if (dev > largest_) largest_ = dev;
    }
    for (int v = 0; v < N; v++) var_[v].push(out[v], limit_for(v, limit_g), probing);
    if (probing && var_[RAW].ready() && --probe_left_ == 0) finished_ = true;
  }

  /** With BoxMonitor::begin_event: the per-event stretches restart. */
  void begin_event() {
    for (auto &v : var_) v.run = v.max = 0;
  }

  void start_probe(int samples) {
    probe_left_ = probe_total_ = samples;
    finished_ = false;
    spikes_ = 0;
    largest_ = 0.0f;
    for (auto &v : var_) v.reset_probe();
  }
  void cancel_probe() { probe_left_ = 0; }
  bool probe_running() const { return probe_left_ > 0; }
  int probe_seconds() const { return probe_total_ / HZ; }
  /** True once, right after a probe has finished; takes each filter's noise as its reference. */
  bool take_result() {
    if (!finished_) return false;
    finished_ = false;
    for (int v = 0; v < N; v++) ref_[v] = var_[v].median_spread(probe_total_);
    return true;
  }

  /** Reference noise per filter (from the last probe), kept across reboots by the YAML. */
  float ref(int v) const { return ref_[v]; }
  void set_ref(int v, float g) { ref_[v] = g; }

  /**
   * Noise test: per filter, the typical 1 s spread / the smallest limit held
   * for `need_s` straight / the median of each 30 s slice's largest spread.
   */
  void noise_summary(char *buf, size_t n, int need_s) const {
    int pos = 0;
    for (int v = 0; v < N && pos < static_cast<int>(n); v++)
      pos += snprintf(buf + pos, n - pos, "%s%s %.2f %.1f %.2f", v ? " | " : "", NAMES[v],
                      var_[v].median_spread(probe_total_), var_[v].min_limit(need_s * HZ),
                      var_[v].median_slice_max());
    if (pos < static_cast<int>(n))
      snprintf(buf + pos, n - pos, " | spikes %d (%.1f g)", spikes_, largest_);
  }
  /** Event: per filter, its longest still stretch and the limit it was held to. */
  void event_summary(char *buf, size_t n, float limit_g) const {
    int pos = 0;
    for (int v = 0; v < N && pos < static_cast<int>(n); v++)
      pos += snprintf(buf + pos, n - pos, "%s%s %.1fs@%.2f", v ? " | " : "", NAMES[v],
                      var_[v].max / static_cast<float>(HZ), limit_for(v, limit_g));
  }

 private:
  static constexpr int HZ = 10;
  static constexpr int WINDOW = 10;
  static constexpr float SPIKE_G = 3.0f;
  static constexpr int LIMITS = 26;  // 0.5 .. 3.0 g
  static constexpr int BINS = 100;
  static constexpr float BIN_G = 0.05f;
  static constexpr int SLICE = 30 * HZ;
  static constexpr int MAX_SLICES = 40;  // 20 min
  static constexpr const char *NAMES[N] = {"raw", "med3", "med5", "avg3", "avg5"};
  static float lim(int i) { return 0.5f + 0.1f * i; }

  struct Variant {
    Ring win{WINDOW};
    int run = 0, max = 0;
    uint16_t hist[BINS] = {};
    uint16_t prun[LIMITS] = {};
    uint16_t pbest[LIMITS] = {};
    float slice_max = 0.0f;
    int slice_n = 0, slices = 0;
    float slice_maxes[MAX_SLICES] = {};

    bool ready() const { return win.size() >= WINDOW; }
    void push(float x, float limit_g, bool probing) {
      win.push(x);
      if (!ready()) return;
      float sd = sqrtf(win.variance());
      run = sd < limit_g ? run + 1 : 0;
      if (run > max) max = run;
      if (!probing) return;
      int bin = sd >= 0.0f && sd < BINS * BIN_G ? static_cast<int>(sd / BIN_G) : BINS - 1;
      hist[bin]++;
      for (int i = 0; i < LIMITS; i++) {
        prun[i] = sd < lim(i) ? prun[i] + 1 : 0;
        if (prun[i] > pbest[i]) pbest[i] = prun[i];
      }
      if (!(sd <= slice_max)) slice_max = sd;
      if (++slice_n == SLICE) {
        if (slices < MAX_SLICES) slice_maxes[slices++] = slice_max;
        slice_max = 0.0f;
        slice_n = 0;
      }
    }
    void reset_probe() {
      for (auto &b : hist) b = 0;
      for (int i = 0; i < LIMITS; i++) prun[i] = pbest[i] = 0;
      slice_max = 0.0f;
      slice_n = slices = 0;
    }
    float median_spread(int total) const {
      int seen = 0;
      for (int b = 0; b < BINS; b++) {
        seen += hist[b];
        if (seen * 2 > total) return (b + 0.5f) * BIN_G;
      }
      return NAN;
    }
    float min_limit(int need) const {
      for (int i = 0; i < LIMITS; i++)
        if (pbest[i] >= need) return lim(i);
      return NAN;
    }
    float median_slice_max() const {
      if (slices == 0) return NAN;
      float s[MAX_SLICES];
      std::copy(slice_maxes, slice_maxes + slices, s);
      std::sort(s, s + slices);
      return slices % 2 ? s[slices / 2] : 0.5f * (s[slices / 2 - 1] + s[slices / 2]);
    }
  };

  float median(int k) const {
    k = std::min(k, have_);
    float s[5];
    std::copy(last_, last_ + k, s);
    std::sort(s, s + k);
    return s[k / 2];
  }
  float mean(int k) const {
    k = std::min(k, have_);
    float t = 0.0f;
    for (int i = 0; i < k; i++) t += last_[i];
    return t / k;
  }
  /** The Still Limit, scaled by how noisy this filter reads next to med3's. */
  float limit_for(int v, float limit_g) const {
    if (std::isnan(ref_[v]) || std::isnan(ref_[MED3]) || ref_[MED3] <= 0.0f) return limit_g;
    return limit_g * ref_[v] / ref_[MED3];
  }

  Variant var_[N];
  float last_[5] = {};
  int have_ = 0;
  float ref_[N] = {NAN, NAN, NAN, NAN, NAN};
  int probe_left_ = 0, probe_total_ = 0, spikes_ = 0;
  bool finished_ = false;
  float largest_ = 0.0f;
};

inline FilterCompare &get_filter_compare() {
  static FilterCompare instance;
  return instance;
}
