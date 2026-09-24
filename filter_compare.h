#pragma once
// Local only, temporary: the box tracker's stillness tests run a second time
// on the unfiltered readings, to see what the median of three buys.

#include "state_analyzer.h"

class FilterCompare {
 public:
  /** Every reading: `raw_g` unfiltered, `med_g` what the box tracker got. */
  void process(float raw_g, float med_g, float limit_g) {
    win_.push(raw_g);
    if (win_.size() < WINDOW) return;
    float sd = sqrtf(win_.variance());
    run_ = sd < limit_g ? run_ + 1 : 0;
    if (run_ > max_) max_ = run_;
    run_plus_ = sd < limit_g + PLUS_G ? run_plus_ + 1 : 0;
    if (run_plus_ > max_plus_) max_plus_ = run_plus_;
    if (probe_left_ > 0) {
      float dev = fabsf(raw_g - med_g);
      if (dev > SPIKE_G) spikes_++;
      if (dev > largest_) largest_ = dev;
      int bin = sd >= 0.0f && sd < BINS * BIN_G ? static_cast<int>(sd / BIN_G) : BINS - 1;
      hist_[bin]++;
      for (int i = 0; i < LIMITS; i++) {
        prun_[i] = sd < limit(i) ? prun_[i] + 1 : 0;
        if (prun_[i] > pbest_[i]) pbest_[i] = prun_[i];
      }
      probe_left_--;
    }
  }

  /** Called with BoxMonitor::begin_event: the per-event stretches restart. */
  void begin_event() { run_ = max_ = run_plus_ = max_plus_ = 0; }
  /** Longest unfiltered stretch this event under the limit, and the limit + 0.5 g. */
  float longest_still_s() const { return max_ / static_cast<float>(HZ); }
  float longest_still_plus_s() const { return max_plus_ / static_cast<float>(HZ); }

  /** Runs alongside BoxMonitor's noise probe, over the same samples. */
  void start_probe(int samples) {
    probe_left_ = probe_total_ = samples;
    spikes_ = 0;
    largest_ = 0.0f;
    for (auto &b : hist_) b = 0;
    for (int i = 0; i < LIMITS; i++) prun_[i] = pbest_[i] = 0;
  }
  void cancel_probe() { probe_left_ = 0; }
  float probe_median_g() const {
    int seen = 0;
    for (int b = 0; b < BINS; b++) {
      seen += hist_[b];
      if (seen * 2 > probe_total_) return (b + 0.5f) * BIN_G;
    }
    return NAN;
  }
  float probe_limit_g(int need_s) const {
    for (int i = 0; i < LIMITS; i++)
      if (pbest_[i] >= need_s * HZ) return limit(i);
    return NAN;
  }
  /** Readings more than SPIKE_G from the median of three, and the largest gap. */
  int spikes() const { return spikes_; }
  float largest_spike_g() const { return largest_; }

 private:
  static constexpr int HZ = 10;
  static constexpr int WINDOW = 10;
  static constexpr float PLUS_G = 0.5f;
  static constexpr float SPIKE_G = 3.0f;
  static constexpr int LIMITS = 21;
  static constexpr int BINS = 100;
  static constexpr float BIN_G = 0.05f;
  static float limit(int i) { return 1.0f + 0.1f * i; }

  Ring win_{WINDOW};
  int run_ = 0, max_ = 0, run_plus_ = 0, max_plus_ = 0;
  int probe_left_ = 0, probe_total_ = 0, spikes_ = 0;
  float largest_ = 0.0f;
  uint16_t hist_[BINS] = {};
  uint16_t prun_[LIMITS] = {};
  uint16_t pbest_[LIMITS] = {};
};

inline FilterCompare &get_filter_compare() {
  static FilterCompare instance;
  return instance;
}
