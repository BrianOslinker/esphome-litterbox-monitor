#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

static const int SA_MAX_SAMPLES = 36000;
/** Long visits can oscillate often; dropping txs breaks post_process vs TS (was 128). */
static const int SA_MAX_TRANSITIONS = 2048;
/** Must cover all tx segments before merge (short visits can exceed 64 transitions). */
static const int SA_MAX_PERIODS = 256;
/** Max RMS windows for median-RMS metric at 10 Hz over SA_MAX_SAMPLES. */
static const int SA_MOTION_HZ = 10;
static const int SA_MAX_MOTION_WINDOWS = SA_MAX_SAMPLES / SA_MOTION_HZ;
static const int SA_MAX_CATS = 5;
/** Zones open on OCCUPIED entry/exit and on delta overflow; a fidgety visit needs room. */
static const int SA_MAX_ZONES = 256;
static const float SA_URINATION_STD_DEV_THRESHOLD_G = 4.0f;
static const int16_t SA_SCALE_ABS = 1;    // 1g, identical to previous behavior
static const int16_t SA_SCALE_DELTA = 10; // 0.1g, used during OCCUPIED/ELIMINATING
/**
 * Samples the buffer keeps in RAM. On the device the rest of a long visit
 * is read back from its journal entry on flash (see WeightBuffer), so this
 * only has to cover what the journal has not written yet: 4096 samples is
 * almost 7 minutes against a 2 KB chunk every ~100 s. Off the device
 * (replay, tests) the whole visit stays in RAM.
 */
#ifndef SA_RING_SAMPLES
#ifdef ESP_PLATFORM
#define SA_RING_SAMPLES 4096
#else
#define SA_RING_SAMPLES SA_MAX_SAMPLES
#endif
#endif
static const int SA_RING = SA_RING_SAMPLES;
static_assert(SA_RING > 0 && SA_RING <= SA_MAX_SAMPLES, "ring size");

/**
 * Scratch for median-RMS over eliminating windows.
 * Function-static so it lives in BSS (not the ESPHome 8 KiB loop stack / heap).
 * Single-threaded: only one finalize/replay at a time.
 */
inline float *sa_motion_rms_scratch() {
  static float buf[SA_MAX_MOTION_WINDOWS];
  return buf;
}

enum class AnalyzerState : uint8_t {
  EMPTY,
  ENTERING,
  OCCUPIED,
  ELIMINATING,
  GAP
};

enum class EliminationType : uint8_t {
  NO_ELIMINATION,
  URINATION,
  DEFECATION,
  BOTH,
  UNKNOWN
};

struct StatePeriod {
  AnalyzerState state;
  int start;
  int end;
  float std_dev;  // negative means undefined
};

struct StateTransition {
  AnalyzerState from;
  AnalyzerState to;
  int index;
};

struct StateResult {
  float cat_weight;
  float waste_weight;
  StatePeriod periods[SA_MAX_PERIODS];
  int period_count;
  EliminationType elimination_type;
  int detected_cat;  // 0-based index, -1 = unknown
};

struct ZoneEntry {
  int start;         // first sample index in this zone
  float baseline_g;  // 0.0 for absolute zones
  int16_t scale;     // SA_SCALE_ABS or SA_SCALE_DELTA
};

// Ring buffer for rolling statistics over a fixed window.
class Ring {
 public:
  Ring() : n_(0), i_(0), filled_(0) { memset(buf_, 0, sizeof(buf_)); }

  explicit Ring(int n) : n_(n), i_(0), filled_(0) {
    memset(buf_, 0, sizeof(buf_));
  }

  void push(float x) {
    buf_[i_] = x;
    i_ = (i_ + 1) % n_;
    if (filled_ < n_) filled_++;
  }

  float mean() const {
    if (!filled_) return 0.0f;
    float s = 0.0f;
    for (int k = 0; k < filled_; k++) s += buf_[k];
    return s / static_cast<float>(filled_);
  }

  // Median of the filled slots (mean of the middle two for an even count).
  float median() const {
    if (!filled_) return 0.0f;
    float tmp[MAX_RING];
    memcpy(tmp, buf_, sizeof(float) * filled_);
    std::sort(tmp, tmp + filled_);
    return filled_ % 2 ? tmp[filled_ / 2] : 0.5f * (tmp[filled_ / 2 - 1] + tmp[filled_ / 2]);
  }

  // Sample variance (divides by n-1)
  float variance() const {
    if (filled_ < 2) return 0.0f;
    float m = mean();
    float s = 0.0f;
    for (int k = 0; k < filled_; k++) {
      float d = buf_[k] - m;
      s += d * d;
    }
    return s / static_cast<float>(filled_ - 1);
  }

  // Checks ALL n slots including unfilled zeros, matching the TS Array.every()
  // behavior where the buffer is pre-filled with zeros.
  template <typename Func>
  bool every(Func fn) const {
    for (int k = 0; k < n_; k++) {
      if (!fn(buf_[k])) return false;
    }
    return true;
  }

  int size() const { return filled_; }

  void reset(int n) {
    n_ = n;
    i_ = 0;
    filled_ = 0;
    memset(buf_, 0, sizeof(buf_));
  }

 private:
  static const int MAX_RING = 16;
  float buf_[MAX_RING];
  int n_;
  int i_;
  int filled_;
};

// Stores raw weight samples for post-hoc per-period variance calculation.
//
// Samples are stored in one of two encodings, tracked via a zone table:
//   Absolute zones (SA_SCALE_ABS=1):  int16_t(weight_g)          -- 1g precision
//   Delta zones (SA_SCALE_DELTA=10):  int16_t((weight_g-base)*10) -- 0.1g precision
//
// Delta zones are opened at each OCCUPIED entry; absolute zones at GAP/ENTERING,
// and whenever a delta zone would overflow (a cat heavier than 3.2 kg leaving
// mid-zone), so no stored sample is ever more than 0.5 g from what was read.
// elimination_motion_metric() decodes per zone and applies TS median-RMS metric.
//
// The buffer is the record of the visit: the analyzer consumes what push()
// returns, so the raw samples plus the zone table replay it exactly.
//
// Storage is a ring of the last SA_RING samples. Older ones are read from
// `backing`, the visit's samples as the journal wrote them (memory-mapped
// flash on the device), which covers the first `durable` samples. A sample
// that leaves the ring before it is durable is lost, and the visit with it:
// complete() goes false and the caller must not analyse or ship it.
class WeightBuffer {
 public:
  void reset() {
    count_ = 0;
    lost_ = false;
    backing_ = nullptr;
    durable_ = 0;
    zone_count_ = 0;
    current_zone_ = -1;
    begin_absolute_zone();
  }

  // Open an absolute (1g) zone starting at the current sample position.
  void begin_absolute_zone() {
    if (zone_count_ < SA_MAX_ZONES)
      zones_[current_zone_ = zone_count_++] = {count_, 0.0f, SA_SCALE_ABS};
  }

  // Open a delta (0.1g) zone with the given baseline starting at current position.
  void begin_delta_zone(float baseline_g) {
    if (zone_count_ < SA_MAX_ZONES)
      zones_[current_zone_ = zone_count_++] = {count_, baseline_g, SA_SCALE_DELTA};
  }

  /**
   * Store a sample and return what it decodes back to — the value the state
   * machine must consume so the buffer is an exact record of the visit.
   * Once the buffer is full nothing is stored and the input passes through;
   * the analyzer stops at MAX_SESSION samples anyway.
   */
  float push(float weight_g) {
    if (count_ >= SA_MAX_SAMPLES || current_zone_ < 0) return weight_g;
    const ZoneEntry *z = &zones_[current_zone_];
    float val = encode_(weight_g, *z);
    if (z->scale != SA_SCALE_ABS && (val > 32767.0f || val < -32768.0f)) {
      // The zone table can be full, in which case the sample clamps below.
      begin_absolute_zone();
      z = &zones_[current_zone_];
      val = encode_(weight_g, *z);
    }
    // Round, don't truncate: a decoded value re-encodes to the same code,
    // so a replay that pushes decoded grams rebuilds this buffer exactly.
    int16_t enc = static_cast<int16_t>(lroundf(std::max(-32768.0f, std::min(32767.0f, val))));
    // The slot about to be reused holds sample count_ - SA_RING.
    if (count_ >= SA_RING && (backing_ == nullptr || count_ - SA_RING >= durable_)) lost_ = true;
    ring_[count_ % SA_RING] = enc;
    count_++;
    return decode_(enc, *z);
  }

  /**
   * Rebuild verbatim from a published record (raw codes plus zone table).
   * `samples` must outlive the buffer when the record is longer than the ring.
   */
  void restore(const int16_t *samples, int n, const ZoneEntry *zones, int nz) {
    count_ = std::min(n, SA_MAX_SAMPLES);
    lost_ = false;
    backing_ = samples;
    durable_ = count_;
    int first = std::max(0, count_ - SA_RING);
    for (int i = first; i < count_; i++) ring_[i % SA_RING] = samples[i];
    zone_count_ = std::min(nz, SA_MAX_ZONES);
    memcpy(zones_, zones, zone_count_ * sizeof(ZoneEntry));
    current_zone_ = zone_count_ - 1;
  }

  /**
   * Where samples that have left the ring can be read, and how many of them
   * are there. Null when there is nowhere (no journal, or it gave up).
   */
  void set_backing(const int16_t *samples, int durable) {
    backing_ = samples;
    durable_ = samples ? durable : 0;
  }

  /** Every sample can still be read: the visit can be analysed and shipped. */
  bool complete() const { return !lost_ && (count_ <= SA_RING || backing_ != nullptr); }

  int count() const { return count_; }
  int zone_count() const { return zone_count_; }
  const ZoneEntry &zone(int i) const { return zones_[i]; }
  int16_t raw_sample(int idx) const {
    if (idx >= count_ - SA_RING) return ring_[idx % SA_RING];
    return backing_ ? backing_[idx] : 0;
  }

  /**
   * Frame bytes still held in RAM, for the journal to copy out: the samples
   * start at frame offset 8 (after "LBV1" and the count). Returns a pointer
   * to the byte at `off` and how many follow it contiguously, or null when
   * that sample has left the ring.
   */
  const uint8_t *frame_bytes(uint32_t off, uint32_t *avail) const {
    *avail = 0;
    if (off < 8 || (off & 1)) return nullptr;
    int s = static_cast<int>((off - 8) / 2);
    if (s >= count_ || s < count_ - SA_RING) return nullptr;
    int pos = s % SA_RING;
    *avail = 2 * static_cast<uint32_t>(std::min(SA_RING - pos, count_ - s));
    return reinterpret_cast<const uint8_t *>(&ring_[pos]);
  }

  /** Grams the analyzer saw for sample `idx`. */
  float sample_to_grams(int idx) const {
    const ZoneEntry *z = zone_for_index(idx);
    if (!z || idx < 0 || idx >= count_) return 0.0f;
    return decode_(raw_sample(idx), *z);
  }

  /**
   * Motion metric for an ELIMINATING period — matches TS processEvent:
   * slice [start+10, end+1-10), decode to grams, eliminatingPeriodMotionMetric(..., hz=10).
   *
   * No heap: decodes window-by-window; RMS list uses sa_motion_rms_scratch() BSS.
   */
  float elimination_motion_metric(int start, int end, int hz) const {
    const int trim = 10;
    int s = start + trim;
    int e = end + 1 - trim;
    if (s >= e || s < 0 || e > count_) return -1.0f;
    int n = e - s;
    if (n < 2) return -1.0f;

    const int ws = hz >= 1 ? hz : 1;
    if (n < ws) return rms_around_mean_range(s, n);

    const int window_count = n / ws;
    if (window_count > SA_MAX_MOTION_WINDOWS) return -1.0f;

    float *rms_values = sa_motion_rms_scratch();
    for (int w = 0; w < window_count; ++w)
      rms_values[w] = rms_around_mean_range(s + w * ws, ws);

    std::sort(rms_values, rms_values + window_count);
    const int mid = window_count / 2;
    if (window_count % 2)
      return rms_values[mid];
    return 0.5f * (rms_values[mid - 1] + rms_values[mid]);
  }

 private:
  const ZoneEntry *zone_for_index(int idx) const {
    for (int z = 0; z < zone_count_; ++z) {
      int next_start = (z + 1 < zone_count_) ? zones_[z + 1].start : count_;
      if (idx >= zones_[z].start && idx < next_start) return &zones_[z];
    }
    return nullptr;
  }

  static float encode_(float weight_g, const ZoneEntry &z) {
    float val = weight_g - z.baseline_g;
    if (z.scale != SA_SCALE_ABS) val *= z.scale;
    return val;
  }

  static float decode_(int16_t enc, const ZoneEntry &z) {
    float val = static_cast<float>(enc);
    if (z.scale == SA_SCALE_ABS) return val;
    return z.baseline_g + val / static_cast<float>(z.scale);
  }

  /** TS rmsAroundMean over decoded grams in [s, s+n) without a heap decode buffer. */
  float rms_around_mean_range(int s, int n) const {
    if (n < 2) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += sample_to_grams(s + i);
    float mean = sum / static_cast<float>(n);
    float sq = 0.0f;
    for (int i = 0; i < n; ++i) {
      float d = sample_to_grams(s + i) - mean;
      sq += d * d;
    }
    return sqrtf(sq / static_cast<float>(n));
  }

  int16_t ring_[SA_RING];
  int count_ = 0;
  bool lost_ = false;
  const int16_t *backing_ = nullptr;
  int durable_ = 0;
  ZoneEntry zones_[SA_MAX_ZONES];
  int zone_count_ = 0;
  int current_zone_ = -1;
};

class StateAnalyzer {
 public:
  // Local: judge one cat by how long it holds still instead of by wobble.
  // Lucy's pees hold still for only 3-9 s at std dev 3-7 g, so the wobble
  // rule missed them or called them poops; her poops held still 14-20 s.
  // For that cat only: a still stretch counts from min_still_s when it left
  // >= 10 g, the longest stretch decides (<= pee_max_s pee, longer poop) and
  // nothing left behind (<= 0 g) is no elimination - lying or standing on
  // the lid reads as long, very still stretches. cat < 0 turns it off.
  // With less than 10 g left a stretch needs min_still_low_s (others: 7 s):
  // her 2026-11-07 10:41 pee held still 6.1 s and left only ~1 g.
  // Kept across init(); the other cats keep the wobble rule.
  void set_duration_rule(int cat, float pee_max_s, float min_still_s, float min_still_low_s = 5.0f) {
    rule_cat_ = cat;
    rule_pee_max_ = static_cast<int>(lroundf(pee_max_s * HZ));
    rule_min_still_ = static_cast<int>(lroundf(min_still_s * HZ));
    rule_min_still_low_ = static_cast<int>(lroundf(min_still_low_s * HZ));
  }

  void init(const float *known_weights_kg, int num_cats) {
    num_known_ = 0;
    for (int i = 0; i < SA_MAX_CATS; i++) {
      slot_g_[i] = 0.0f;
      cat_presence_[i] = 0;
    }
    for (int i = 0; i < num_cats && i < SA_MAX_CATS; i++) {
      if (known_weights_kg[i] > 0.0f) {
        slot_g_[i] = known_weights_kg[i] * 1000.0f;
        known_g_[num_known_++] = slot_g_[i];
      }
    }
    std::sort(known_g_, known_g_ + num_known_);
    reset();
  }

  void attach_buffer(WeightBuffer *buf) { wbuf_ = buf; }

  /** `weight` in grams (float — matches firmware HX711 path). */
  void process_sample(float weight, int index) {
    // Stop updating rings/state once the weight buffer is full (1 h at 10 Hz).
    if (session_active_ && index - session_start_ > MAX_SESSION) return;

    current_weight_g_ = weight;
    current_sample_ = index;
    // Local: drop single garbled HX711 reads. One under the 4 kg spike filter
    // (−2 kg on 2026-11-06 19:19; 1–3 a day) breaks a still stretch for a
    // second, enough to cut a 5–8 s squat short. Judge each reading one
    // sample late: if it sits far from both neighbours while they agree, use
    // their mean instead. Only true one-sample glitches change — a plain
    // median of three also smooths a cat's wobble, makes it look still for
    // longer and shifted 6 of 181 replayed verdicts (thresholds were tuned
    // on raw readings). The buffer still records the raw reading, so replays
    // filter the same way. 100 ms of delay.
    {
      float g = weight;
      if (std::isnan(med_d1_)) {
        weight = g;
      } else {
        weight = med_d1_;
        if (!std::isnan(med_d2_) && std::abs(g - med_d2_) <= GLITCH_PAIR_G &&
            std::abs(med_d1_ - 0.5f * (g + med_d2_)) > GLITCH_G)
          weight = 0.5f * (g + med_d2_);
      }
      med_d2_ = med_d1_;
      med_d1_ = g;
    }
    // Everything below is measured from what sat on the box just before the
    // cat got in. current_weight_g_ stays raw: the weight buffer's zones are.
    track_pre_entry(weight);
    weight -= pre_entry_g_;
    window_.push(weight);
    weight_hist_.push(weight);
    float mean1s = window_.mean();
    mean_hist_.push(mean1s);
    float var10 = weight_hist_.variance();
    float var10sample = var10 > 0.0f ? sqrtf(var10) : 0.0f;
    bool stable_now = var10sample > 0.0f && var10sample < STABLE_VARIANCE_SQRT;

    float entry_delta = entry_threshold();
    float presence_th =
        cat_weight_ > 0.0f ? cat_weight_ * PRESENCE_FRAC : entry_delta;

    switch (state_) {
      case AnalyzerState::EMPTY:
        if (mean1s > entry_delta) {
          start_session();
          transition_to(AnalyzerState::ENTERING);
        }
        break;

      case AnalyzerState::ENTERING:
        if (!confirm_presence(mean1s)) {
          if (mean1s < 0.5 * entry_delta)
            transition_to(AnalyzerState::GAP, WINDOW / 2);
          break;
        }
        if (mean_hist_.variance() < 10.0f &&
            mean_hist_.every(
                [this](float m) { return near_known(m); })) {
          transition_to(AnalyzerState::OCCUPIED, WINDOW / 2);
        } else {
          stable_cnt_ = 0;
        }
        break;

      case AnalyzerState::OCCUPIED: {
        int ci = closest_known_cat(mean1s);
        if (ci >= 0) cat_presence_[ci]++;
        if (stable_now && near_known(mean1s))
          transition_to(AnalyzerState::ELIMINATING);
        if (weight < entry_delta) {
          transition_to(AnalyzerState::GAP, WINDOW);
          exit_below_ = 0;
          break;
        }
        if (cat_weight_ > 0.0f &&
            cat_weight_ - mean1s > presence_th) {
          if (++exit_below_ >= EXIT_HOLD) {
            exit_below_ = 0;
            gap_cnt_ = 0;
            transition_to(AnalyzerState::ENTERING, WINDOW);
          }
        } else {
          exit_below_ = 0;
        }
        break;
      }

      case AnalyzerState::ELIMINATING: {
        int ci_elim = closest_known_cat(mean1s);
        if (ci_elim >= 0) cat_presence_[ci_elim]++;
        if (stable_now) {
          stable_cnt_++;
          elim_sum_ += mean1s;
          elim_count_++;
          cat_weight_ = elim_sum_ / static_cast<float>(elim_count_);
        } else {
          if (elim_count_ > best_elim_dur_) {
            best_elim_dur_ = elim_count_;
            best_elim_weight_ = elim_sum_ / static_cast<float>(elim_count_);
          }
          elim_sum_ = 0.0f;
          elim_count_ = 0;
          stable_cnt_ = 0;
          transition_to(AnalyzerState::OCCUPIED);
        }
        if (cat_weight_ > 0.0f &&
            cat_weight_ - mean1s > presence_th) {
          if (++exit_below_ >= EXIT_HOLD) {
            exit_below_ = 0;
            gap_cnt_ = 0;
            transition_to(AnalyzerState::GAP);
          }
        } else {
          exit_below_ = 0;
        }
        break;
      }

      case AnalyzerState::GAP:
        gap_cnt_++;
        if (mean1s > entry_delta) {
          if (stable_now) {
            if (near_known(mean1s))
              transition_to(AnalyzerState::ELIMINATING);
          } else {
            transition_to(AnalyzerState::ENTERING);
          }
        } else if (gap_cnt_ > REENTRY_WIN) {
          // Median of the last second: one garbled HX711 read must not
          // decide the waste (6 Nov 19:19, a -2010 g read turned a pee
          // into -184 g and no elimination).
          waste_weight_ = window_.median();
          waste_set_ = true;
          return;
        }
        break;
    }
  }

  // Call after the session ends to compute final results.
  // Requires the WeightBuffer that was filled alongside process_sample calls.
  //
  // Returns a reference to member storage — do NOT put StateResult on the
  // ESPHome loop stack (periods[SA_MAX_PERIODS] alone is ~4 KiB).
  const StateResult &finalize(const WeightBuffer &buf,
                              const float *known_weights_kg,
                              int num_cats,
                              float std_dev_threshold = SA_URINATION_STD_DEV_THRESHOLD_G) {
    float best_w = best_elim_weight_;
    if (elim_count_ > best_elim_dur_ && elim_count_ > 0)
      best_w = elim_sum_ / static_cast<float>(elim_count_);
    result_.cat_weight = (best_w > 0.0f) ? best_w : cat_weight_;
    // The event usually closes a few seconds after the cat leaves, well
    // before REENTRY_WIN, so the GAP branch never gets to set the waste.
    // What the scale settled on over the last second of that gap is the
    // same thing end_event tares away, so take it from there (its median,
    // so a single glitched read can't swing it).
    if (state_ == AnalyzerState::GAP && current_sample_ - state_start_ >= WINDOW) {
      waste_weight_ = window_.median();  // not mean: see the GAP branch
      waste_set_ = true;
    }
    // No cat, nothing to measure from: report against the tare, as before.
    if (!cat_seen_ && waste_set_) waste_weight_ += pre_entry_g_;
    result_.waste_weight = waste_weight_;
    result_.period_count = 0;
    result_.elimination_type = EliminationType::UNKNOWN;
    result_.detected_cat = -1;

    // Local: the cat is known before the periods are judged, so a cat with
    // its own duration rule (set_duration_rule) can be judged by it.
    const int cat = detected_cat_from_presence();
    const bool duration_rule = rule_cat_ >= 0 && cat == rule_cat_;

    post_process(result_.periods, result_.period_count, duration_rule);

    for (int i = 0; i < result_.period_count; i++) {
      result_.periods[i].std_dev =
          buf.elimination_motion_metric(result_.periods[i].start, result_.periods[i].end, HZ);
    }

    result_.elimination_type =
        duration_rule
            ? classify_by_duration(result_.periods, result_.period_count)
            : classify_elimination(result_.periods, result_.period_count, std_dev_threshold);

    result_.detected_cat = cat;

    return result_;
  }

  void reset() {
    state_ = AnalyzerState::EMPTY;
    state_start_ = 0;
    session_active_ = false;
    window_.reset(WINDOW);
    weight_hist_.reset(WINDOW);
    mean_hist_.reset(3);
    cat_weight_ = 0.0f;
    elim_sum_ = 0.0f;
    elim_count_ = 0;
    best_elim_weight_ = 0.0f;
    best_elim_dur_ = 0;
    exit_below_ = 0;
    gap_cnt_ = 0;
    stable_cnt_ = 0;
    tx_count_ = 0;
    waste_weight_ = 0.0f;
    for (int i = 0; i < SA_MAX_CATS; i++) cat_presence_[i] = 0;
    pre_entry_g_ = 0.0f;
    pre_win_.reset(WINDOW);
    pre_still_ = 0;
    cat_seen_ = false;
    waste_set_ = false;
    med_d1_ = NAN;
    med_d2_ = NAN;
  }

  AnalyzerState current_state() const { return state_; }

  /**
   * Level the visit was measured from, relative to the event's tare: the
   * last still plateau before any cat was seen. Non-zero when something was
   * put on the box, or the box settled, after the event began (the lid going
   * back on just before a cat jumps in). Waste should be taken from it too.
   */
  float pre_entry_g() const { return pre_entry_g_; }

 private:
  static constexpr int HZ = 10;
  /** sqrt(250) — TS compares sqrt(sample var) to this (not variance to 250). */
  static constexpr float STABLE_VARIANCE_SQRT = 15.811388f;
  // Local: 3 s (upstream 1.5 s). Osiris's poop on 2026-11-02 02:18 held
  // still for 1-4 s at a time with 2-3 s jolts between, so no piece reached
  // the 7 s minimum. 2.5-5 s changes no other journaled visit.
  static constexpr float STABLE_MERGE_GAP = 3.0f * HZ;
  // A merged period that moves this much is not a cat holding still (real
  // eliminating periods measure < 10 g); without it a 3 s gap bridged a
  // 5.8 s, 158 g period into Lucy's 2026-10-28 visit and made it "unknown".
  static constexpr float ELIM_MAX_STD_DEV = 20.0f;
  static constexpr float ENTRY_DELTA_MIN = 1200.0f;
  static constexpr float ENTRY_DELTA_FRAC = 0.22f;
  static constexpr float PRESENCE_FRAC = 0.28f;
  static constexpr int EXIT_HOLD = 6;
  static constexpr int REENTRY_WIN = 15 * HZ;
  /** At 10 Hz, 36000 samples = 1 h — matches SA_MAX_SAMPLES. */
  static constexpr int MAX_SESSION = 60 * 60 * HZ;
  static constexpr float KNOWN_TOL = 0.1f;
  static constexpr int WINDOW = 10;
  // set_duration_rule; off by default.
  int rule_cat_ = -1;
  int rule_pee_max_ = 10 * HZ;
  int rule_min_still_ = 5 * HZ;
  int rule_min_still_low_ = 5 * HZ;
  /**
   * A pre-entry plateau: 1 s spread under PRE_STILL_G, held PRE_HOLD, within
   * PRE_MAX_G of the tare. The lid held 0.5-1.6 g; a cat holding part of its
   * weight on the box wobbles far more. The cap keeps a cat's paws out of it:
   * a baseline taken under them would push the cat outside its match window
   * and lose the visit, which is worse than a wrong weight.
   */
  static constexpr float PRE_STILL_G = 2.0f;
  static constexpr int PRE_HOLD = 3 * HZ;
  static constexpr float PRE_MAX_G = 600.0f;

  AnalyzerState state_ = AnalyzerState::EMPTY;
  float known_g_[SA_MAX_CATS] = {};
  /** Known cat weight per config slot (g); 0 = empty. Same indexing as init() input (sorted in replay). */
  float slot_g_[SA_MAX_CATS] = {};
  int cat_presence_[SA_MAX_CATS] = {};
  int num_known_ = 0;
  Ring window_{WINDOW};
  Ring weight_hist_{WINDOW};
  Ring mean_hist_{3};
  int exit_below_ = 0;
  int gap_cnt_ = 0;
  int stable_cnt_ = 0;
  bool session_active_ = false;
  int session_start_ = 0;
  int current_sample_ = 0;
  int state_start_ = 0;  // sample on which state_ was entered (not backdated)
  float waste_weight_ = 0.0f;
  float cat_weight_ = 0.0f;
  float elim_sum_ = 0.0f;
  int elim_count_ = 0;
  float best_elim_weight_ = 0.0f;
  int best_elim_dur_ = 0;
  float current_weight_g_ = 0.0f;
  float pre_entry_g_ = 0.0f;
  Ring pre_win_{WINDOW};
  int pre_still_ = 0;
  bool cat_seen_ = false;  // OCCUPIED reached: the baseline is frozen
  bool waste_set_ = false;
  float med_d1_ = NAN;  // previous two raw readings, for the glitch check
  float med_d2_ = NAN;
  // A glitch: the reading sits over GLITCH_G from its neighbours' mean while
  // they agree within GLITCH_PAIR_G. Cat jolts span several samples.
  static constexpr float GLITCH_G = 250.0f;
  static constexpr float GLITCH_PAIR_G = 40.0f;
  WeightBuffer *wbuf_ = nullptr;
  StateTransition txs_[SA_MAX_TRANSITIONS];
  int tx_count_ = 0;
  /** Finalize output — BSS, not loop-stack. */
  StateResult result_{};
  /**
   * Shared period scratch for post_process then classify_elimination
   * (sequential; never live at the same time). Keeps ~4 KiB off the stack.
   */
  StatePeriod period_scratch_[SA_MAX_PERIODS] = {};

  float entry_threshold() const {
    float min_known = num_known_ > 0 ? known_g_[0] : 0.0f;
    return min_known > 0.0f
               ? std::max(ENTRY_DELTA_MIN, min_known * ENTRY_DELTA_FRAC)
               : ENTRY_DELTA_MIN;
  }

  void start_session() {
    session_active_ = true;
    session_start_ = current_sample_;
    cat_weight_ = 0.0f;
    elim_sum_ = 0.0f;
    elim_count_ = 0;
    best_elim_weight_ = 0.0f;
    best_elim_dur_ = 0;
    exit_below_ = 0;
    gap_cnt_ = 0;
    stable_cnt_ = 0;
    tx_count_ = 0;
    for (int i = 0; i < SA_MAX_CATS; i++) cat_presence_[i] = 0;
  }

  void transition_to(AnalyzerState ns, int offset = 0) {
    if (state_ == ns) return;
    if (ns == AnalyzerState::OCCUPIED || ns == AnalyzerState::ELIMINATING) cat_seen_ = true;
    if (session_active_ && ns == AnalyzerState::EMPTY) {
      state_ = ns;
      session_active_ = false;
      return;
    }
    if (tx_count_ < SA_MAX_TRANSITIONS)
      txs_[tx_count_++] = {state_, ns, current_sample_ - offset};
    if (wbuf_) {
      if (ns == AnalyzerState::OCCUPIED && state_ != AnalyzerState::ELIMINATING) {
        wbuf_->begin_delta_zone(current_weight_g_);
      } else if ((ns == AnalyzerState::GAP || ns == AnalyzerState::ENTERING) &&
                 (state_ == AnalyzerState::OCCUPIED ||
                  state_ == AnalyzerState::ELIMINATING)) {
        wbuf_->begin_absolute_zone();
      }
    }
    state_ = ns;
    state_start_ = current_sample_;
  }

  void track_pre_entry(float raw_g) {
    if (cat_seen_) return;
    pre_win_.push(raw_g);
    if (pre_win_.size() < WINDOW || sqrtf(pre_win_.variance()) >= PRE_STILL_G) {
      pre_still_ = 0;
      return;
    }
    float m = pre_win_.mean();
    if (++pre_still_ >= PRE_HOLD && std::abs(m) <= PRE_MAX_G) pre_entry_g_ = m;
  }

  /** among cats within tolerance, smallest absolute diff wins. */
  int closest_known_cat(float val_g, float tol = KNOWN_TOL) const {
    int best = -1;
    float min_diff = 1e9f;
    for (int i = 0; i < SA_MAX_CATS; i++) {
      float w = slot_g_[i];
      if (w <= 0.0f) continue;
      float diff = std::abs(val_g - w);
      if (diff <= w * tol && diff < min_diff) {
        min_diff = diff;
        best = i;
      }
    }
    return best;
  }

  /** highest count wins; ties go to lower slot index. */
  int detected_cat_from_presence() const {
    int best = -1;
    int best_count = 0;
    for (int i = 0; i < SA_MAX_CATS; i++) {
      if (slot_g_[i] <= 0.0f) continue;
      if (cat_presence_[i] > best_count) {
        best_count = cat_presence_[i];
        best = i;
      }
    }
    return best_count > 0 ? best : -1;
  }

  bool near_known(float val, float tol = KNOWN_TOL) const {
    for (int i = 0; i < num_known_; i++) {
      float w = known_g_[i];
      if (w > 0.0f && std::abs(val - w) / w <= tol) return true;
    }
    return false;
  }

  bool confirm_presence(float rel) const {
    return near_known(rel) || rel > entry_threshold();
  }

  void post_process(StatePeriod *out, int &count, bool duration_rule = false) {
    count = 0;
    if (tx_count_ == 0) return;

    StatePeriod *tmp = period_scratch_;
    int n = 0;

    for (int i = 0; i < tx_count_ && n < SA_MAX_PERIODS; i++) {
      int s = (i == 0) ? session_start_ : txs_[i].index;
      int e = (i + 1 < tx_count_) ? txs_[i + 1].index : current_sample_;
      AnalyzerState st = txs_[i].to;
      if (st != AnalyzerState::EMPTY && e > s)
        tmp[n++] = {st, s, e, -1.0f};
    }

    // Merge short OCCUPIED gaps between ELIMINATING periods
    for (int i = 1; i < n - 1; i++) {
      if (tmp[i - 1].state == AnalyzerState::ELIMINATING &&
          tmp[i].state == AnalyzerState::OCCUPIED &&
          tmp[i + 1].state == AnalyzerState::ELIMINATING &&
          (tmp[i].end - tmp[i].start) < (int)STABLE_MERGE_GAP &&
          (tmp[i - 1].end - tmp[i - 1].start) > HZ &&
          (tmp[i + 1].end - tmp[i + 1].start) > HZ) {
        tmp[i - 1].end = tmp[i + 1].end;
        for (int j = i; j < n - 2; j++) tmp[j] = tmp[j + 2];
        n -= 2;
        i--;
      }
    }

    // Downgrade short ELIMINATING periods. 8 s matches cat-health's
    // StateAnalyzer (re-tuned 2026-08-19 on 1818 visits); at 5 s a cat
    // holding still for a few seconds while exploring read as an elimination.
    // Local: 7 s. Lucy's small pee on 2026-10-24 held still for 7.0 s and
    // was dropped at 8 s; replaying the journal at 7 s changed no other visit.
    // A 5-7 s period is kept when the cat left weight behind: Lucy's visit on
    // 2026-10-26 held still for 6.8 s and left +37 g. Exploring cats that hold
    // still ~5 s leave nothing, so they still read as no elimination.
    // The duration-rule cat: rule_min_still_low_ (5 s) - see set_duration_rule.
    const int min_elim = duration_rule ? rule_min_still_low_ : 7 * HZ;
    const int min_elim_with_waste = duration_rule ? rule_min_still_ : 5 * HZ;
    const float min_elim_waste_g = 10.0f;
    for (int i = 0; i < n; i++) {
      int len = tmp[i].end - tmp[i].start;
      if (tmp[i].state == AnalyzerState::ELIMINATING && len < min_elim &&
          !(len >= min_elim_with_waste && waste_weight_ >= min_elim_waste_g))
        tmp[i].state = AnalyzerState::OCCUPIED;
    }

    if (n == 0) return;

    // Merge consecutive same-state periods
    out[0] = tmp[0];
    count = 1;
    for (int i = 1; i < n; i++) {
      if (tmp[i].state == out[count - 1].state) {
        out[count - 1].end = tmp[i].end;
      } else if (count < SA_MAX_PERIODS) {
        out[count++] = tmp[i];
      }
    }
  }

  EliminationType classify_elimination(const StatePeriod *p,
                                       int count,
                                       float threshold) {
    StatePeriod *elim = period_scratch_;
    int ec = 0;
    for (int i = 0; i < count; i++) {
      if (p[i].state == AnalyzerState::ELIMINATING && p[i].std_dev >= 0.0f &&
          p[i].std_dev < ELIM_MAX_STD_DEV)
        elim[ec++] = p[i];
    }

    if (ec == 0) return EliminationType::NO_ELIMINATION;
    if (ec == 1)
      return elim[0].std_dev < threshold
                 ? EliminationType::URINATION
                 : EliminationType::DEFECATION;
    if (ec == 2) {
      bool a_uri = elim[0].std_dev < threshold;
      bool b_uri = elim[1].std_dev < threshold;
      if (a_uri != b_uri) return EliminationType::BOTH;
    }
    return EliminationType::UNKNOWN;
  }

  // Local: set_duration_rule's verdict - the longest eliminating stretch.
  EliminationType classify_by_duration(const StatePeriod *p, int count) const {
    if (waste_weight_ <= 0.0f) return EliminationType::NO_ELIMINATION;
    int longest = 0;
    for (int i = 0; i < count; i++)
      if (p[i].state == AnalyzerState::ELIMINATING)
        longest = std::max(longest, p[i].end - p[i].start);
    if (longest == 0) return EliminationType::NO_ELIMINATION;
    return longest <= rule_pee_max_ ? EliminationType::URINATION : EliminationType::DEFECATION;
  }

};

// ---------------------------------------------------------------------------
// BoxMonitor — what is on the scale besides a cat.
//
// Runs on the absolute (calibrated, un-tared) weight at 10 Hz, always, not only
// during activity: the interesting moments straddle event boundaries (the box
// leaves in one event and comes back in the next). Absolute grams make the
// shapes unambiguous where event-relative deltas are not:
//   raw < -lift_g        the whole monitor is in the air, frame hanging
//   |raw| < off_tol      bare board — the box is off, and 0 g is a known weight
//   raw ≈ box_g          the box came back empty: a deep clean
//   then +N kg steps     bags of litter poured in, evened out, left alone
// Per event it also keeps the list of plateaus (stable ≥ 2 s) so the end-of-
// event classifier can ask "rose monotonically, then went inert?".
//
// A removable lid (lid_g > 0). Nothing a cat does takes the box below where it
// stood with the lid on, so the lid is off once a level holds a lid's weight
// below that (or one step of about its weight goes down). It is back on when
// the level rises after a hard press — seating a lid pushes 2-8 kg for a
// moment, a cat landing overshoots its weight by about 1 kg at most — and
// either the rise is about its weight or the box then goes dead still.
// Litter pours in as a ramp with no press. Nothing is decided with a cat on
// the box. Levels measured with the lid off count as if it were on, and the
// state carries over between events, like absence. While the lid is off the
// event it came off in is held open (up to lid_hold_s), so a cleaning is
// judged on where it started and ended.
// ---------------------------------------------------------------------------

static const int BM_MAX_PLATEAUS = 32;

enum class BoxState : uint8_t { NORMAL, LIFTED, OFF, EMPTY };

enum class BoxEventKind : uint8_t { NONE, LIFTED, DEEP_CLEAN, TOP_UP, SCOOP };

struct Plateau {
  int start;
  int end;  // exclusive
  float mean_g;
  float sigma_g;
  bool lid_off;  // the lid was off the box for this plateau
};

struct BoxConfig {
  float box_g;         // configured empty box weight; 0 = unknown (no deep-clean detection)
  float off_tol_g;     // |raw| below this, stable: box off (and zero check)
  float empty_tol_g;   // |raw - box_g| below this, stable, after absence: deep clean
  float lift_g;        // raw below -lift_g: monitor lifted
  float return_tol_g;  // put back within this of the pre-event level: nothing changed
  float top_up_min_g;  // smallest rise reported as a top-up
  float scoop_min_g;   // drop (positive number) that counts as a scoop
  int scoop_min_s;     // ...unless the event is at least this long
  int settle_s;        // inert seconds before a top-up ends the event
  // 1 s spread below which the box is inert: nothing alive on it. Over 1552
  // verified visits the longest a cat held a 1 s window under 1 g was 11 s;
  // poured litter holds it forever. A noisier board needs a higher limit.
  float inert_sigma_g = 1.0f;
  float lid_g = 0.0f;  // weight of a removable lid; 0 = no lid, nothing tracked
  float lid_tol_g = 30.0f;  // a step within this of lid_g is the lid
  int lid_hold_s = 240;     // longest an event is held open with the lid off
};

struct BoxEvent {
  BoxEventKind kind;
  float level_g;         // absolute level at the end (last plateau if there is one)
  float litter_added_g;  // deep clean / top-up: how much went in
  float box_measured_g;  // deep clean: what the empty box actually weighed
  float zero_error_g;    // bare-board reading, valid when zero_valid
  bool zero_valid;
  bool absent;           // box not on the board at the end — do not tare
  bool scooped;          // top-up: litter came out before the bag went in
  float longest_still_s;  // longest inert stretch with something on the box
  bool lid_off;          // the lid is off at the end (level_g counts it anyway)
  float lid_delta_g;     // what the lid did to the weight: +lid_g put back, -lid_g taken off
  bool lid_healed;       // it was taken for off, but the box was back at its lid-on level
};

class BoxMonitor {
 public:
  void configure(const BoxConfig &c) {
    cfg_ = c;
    // No lid (any more): nothing can be off, and nothing carries over.
    if (cfg_.lid_g <= 0.0f) lid_off_ = pl_lid_off_ = ev_lid_off_ = lid_opened_ = lid_on_pending_ = false;
  }
  const BoxConfig &config() const { return cfg_; }

  /** `base_g`: absolute level the event is measured against (the last tare). */
  void begin_event(float base_g) {
    base_g_ = base_g;
    ev_start_ = n_;
    pl_count_ = 0;
    pl_overflow_ = false;
    saw_lifted_ = false;
    saw_off_ = false;
    began_absent_ = absent_;
    deep_clean_ = false;
    empty_pl_ = -1;
    zero_valid_ = false;
    zero_error_g_ = 0.0f;
    box_measured_g_ = 0.0f;
    inert_len_ = 0;
    inert_max_ = 0;
    ev_lid_off_ = lid_off_;
    // Only a lid that comes off or goes on in this event waives the scoop
    // minimum: one left off from before says nothing about this event (8 Nov:
    // with the lid taken for off, 8 s sags of 50-150 g were scoops).
    lid_opened_ = false;
    lid_on_step_g_ = cfg_.lid_g;
    // With the lid on, the tare is where the box stands: anything a lid's
    // weight below it is the lid off. With it off, keep the level it left.
    if (!lid_off_) lid_ref_g_ = base_g;
    hold_broken_ = false;
    lid_back_ = false;
    lid_healed_ = false;
  }

  /**
   * Scale noise probe: for `samples` readings (box left untouched) record the
   * 1 s spread, and its largest value in each settle_s slice, to size
   * inert_sigma_g for this board.
   */
  void start_noise_probe(int samples) {
    probe_left_ = probe_total_ = samples;
    probe_finished_ = false;
    for (auto &b : probe_hist_) b = 0;
    probe_slices_ = probe_slice_n_ = 0;
    probe_slice_max_ = 0.0f;
  }
  void cancel_noise_probe() { probe_left_ = 0; }
  bool noise_probe_running() const { return probe_left_ > 0; }
  /** True once, right after a probe has finished. */
  bool take_noise_probe_result() {
    if (!probe_finished_) return false;
    probe_finished_ = false;
    return true;
  }
  /** Median 1 s spread over the last probe, in grams. */
  float noise_probe_median_g() const {
    int seen = 0;
    for (int b = 0; b < PROBE_BINS; b++) {
      seen += probe_hist_[b];
      if (seen * 2 > probe_total_) return (b + 0.5f) * PROBE_BIN_G;
    }
    return NAN;
  }
  /**
   * The limit the untouched box needs to hold still for settle_s: the largest
   * 1 s spread of a typical slice that long (the median over the probe's
   * slices). Steadier from run to run than asking for
   * one unbroken stretch, which a single jolt can break. NAN if the probe was
   * shorter than one slice.
   */
  float noise_probe_min_limit_g() const {
    if (probe_slices_ == 0) return NAN;
    float s[PROBE_MAX_SLICES];
    std::copy(probe_maxes_, probe_maxes_ + probe_slices_, s);
    std::sort(s, s + probe_slices_);
    int k = probe_slices_ / 2;
    return probe_slices_ % 2 ? s[k] : 0.5f * (s[k - 1] + s[k]);
  }

  /**
   * The lid came off during this event and is still off: keep the event open
   * so the cleaning is judged as a whole, unless a cat got on or it has been
   * off for lid_hold_s.
   */
  bool holding() const {
    return cfg_.lid_g > 0.0f && lid_off_ && lid_off_n_ >= ev_start_ && !absent_ && !hold_broken_ &&
           n_ - lid_off_n_ < cfg_.lid_hold_s * HZ;
  }

  /**
   * Every sample, in absolute grams, event or not. `raw`: the same reading
   * before any smoothing, if `g` is smoothed (a lid press lasts a few samples).
   */
  void process(float g, float raw = NAN) {
    n_++;
    win_.push(g);
    // Between levels: the hardest press, and how long it weighed like a cat.
    trans_max_ = std::max(trans_max_, std::isnan(raw) ? g : std::max(g, raw));
    heavy_run_ = !std::isnan(box_level_g_) && g > box_level_g_ + CAT_PRESS_G ? heavy_run_ + 1 : 0;
    trans_heavy_ = std::max(trans_heavy_, heavy_run_);
    if (win_.size() < WINDOW) return;
    float m = win_.mean();
    float sd = sqrtf(win_.variance());
    bool stable = sd < PLATEAU_SIGMA;
    inert_len_ = sd < cfg_.inert_sigma_g ? inert_len_ + 1 : 0;
    // A pressed-on lid not yet confirmed: the box must go dead still soon.
    if (lid_on_pending_) {
      if (pl_n_ == 0 || n_ > lid_on_deadline_) {
        lid_on_pending_ = false;
      } else if (inert_len_ > 0) {
        lid_on_pending_ = false;
        set_lid_on(lid_on_pending_d_);
      }
    }

    if (stable) {
      if (pl_n_ == 0) {
        pl_start_ = n_ - WINDOW;
        pl_lid_off_ = lid_off_;
      }
      pl_n_++;
      double d = g - pl_mean_;
      pl_mean_ += d / pl_n_;
      pl_m2_ += d * (g - pl_mean_);
      // The lid is judged once per plateau, on its first second that holds
      // really still: hands resting on the box make looser plateaus.
      if (!lid_level_ok_ && pl_n_ >= LID_HOLD && sd < LID_LEVEL_SIGMA) {
        lid_level_ok_ = true;
        track_lid(m);
      }
    } else if (pl_n_ > 0) {
      close_plateau();
    }
    // Only while something (a cat, poured litter) is on the box: the quiet
    // after a cat has left, before the event closes, is the box, not the cat.
    if (inert_len_ > inert_max_ && m - base_g_ >= cfg_.top_up_min_g) inert_max_ = inert_len_;
    if (probe_left_ > 0) {
      // Top bin for anything off the scale, NAN included.
      int bin = sd >= 0.0f && sd < PROBE_BINS * PROBE_BIN_G
                    ? static_cast<int>(sd / PROBE_BIN_G) : PROBE_BINS - 1;
      probe_hist_[bin]++;
      if (!(sd <= probe_slice_max_)) probe_slice_max_ = sd;
      if (++probe_slice_n_ >= std::max(1, cfg_.settle_s) * HZ) {
        if (probe_slices_ < PROBE_MAX_SLICES) probe_maxes_[probe_slices_++] = probe_slice_max_;
        probe_slice_max_ = 0.0f;
        probe_slice_n_ = 0;
      }
      if (--probe_left_ == 0) probe_finished_ = true;
    }

    // Box state. Absence latches until a stable plateau with something in
    // the box shows up again, so "off in one event, back in the next" works.
    bool absent_now = below_box(m);
    if (absent_now) ref_stale_ = true;
    if (m < -cfg_.lift_g) {
      state_ = BoxState::LIFTED;
      saw_lifted_ = true;
      absent_ = true;
      hold_ = 0;
    } else if (stable && std::abs(m) < cfg_.off_tol_g) {
      state_ = BoxState::OFF;
      saw_off_ = true;
      absent_ = true;
      if (++hold_ >= HOLD && pl_n_ >= HOLD) {
        zero_error_g_ = static_cast<float>(pl_mean_);
        zero_valid_ = true;
      }
    } else if (stable && cfg_.box_g > 0.0f && (absent_ || state_ == BoxState::EMPTY) &&
               (std::abs(m - cfg_.box_g) < cfg_.empty_tol_g || empty_without_lid(m))) {
      if (state_ != BoxState::EMPTY && ++hold_ >= HOLD) {
        state_ = BoxState::EMPTY;
        deep_clean_ = true;
        // An empty box can come back with its lid on or off.
        bool no_lid = std::abs(m - cfg_.box_g) >= cfg_.empty_tol_g;
        lid_off_ = pl_lid_off_ = no_lid;
        if (no_lid) lid_opened_ = true;
        lid_off_filled_ = false;
        lid_ref_g_ = with_lid(static_cast<float>(pl_mean_), no_lid);
        last_level_g_ = box_level_g_ = static_cast<float>(pl_mean_);
        ref_stale_ = false;
        box_measured_g_ = with_lid(static_cast<float>(pl_mean_), no_lid);
        empty_pl_ = pl_count_;  // the open plateau closes into this slot
        absent_ = false;
      }
    } else {
      if (absent_now) absent_ = true;
      else if (stable) absent_ = false;
      if (state_ == BoxState::LIFTED || state_ == BoxState::OFF) {
        if (!absent_now) state_ = BoxState::NORMAL;
      } else if (state_ == BoxState::EMPTY && stable) {
        state_ = BoxState::NORMAL;
      }
      hold_ = 0;
    }
  }

  BoxState state() const { return state_; }
  bool absent() const { return absent_; }
  bool lid_off() const { return lid_off_; }
  /**
   * Once, right after the lid went back on in the event it came off in, with
   * nothing added: the cleaning can end now rather than run into a cat that
   * jumps in a moment later. (A top-up still waits to settle.)
   */
  bool take_lid_back() {
    bool r = lid_back_;
    lid_back_ = false;
    return r;
  }

  /**
   * True once the stability window is full and `state()` means something. Until
   * then it is only the NORMAL it was constructed with, which is a guess: a
   * board that boots with the box off reads normal for the first second.
   */
  bool ready() const { return win_.size() >= WINDOW; }

  /**
   * True once a top-up (or a deep clean plus refill) has gone inert for
   * settle_s: the box is settled and the event can end without waiting for the
   * inactivity timeout. Inertness alone is the proof there is no cat — a bag
   * can weigh exactly what a cat does, but no cat holds σ < 1 g for 30 s.
   */
  bool settled() const {
    if (inert_len_ < cfg_.settle_s * HZ || absent_) return false;
    return deep_clean_ || with_lid(current_level(), lid_off_) - ev_base() >= cfg_.top_up_min_g;
  }

  /**
   * Classify the event. Call once, at event end, before taring.
   * `cat_event`: the analyzer saw a known cat during this event. Only scoops
   * defer to it; a rise that went inert is litter whatever the analyzer
   * matched it to, and the caller drops the visit.
   */
  const BoxEvent &finalize(int duration_s, bool cat_event) {
    bool ended_stable = pl_n_ >= MIN_PLATEAU;
    bool ended_inert = inert_len_ >= cfg_.settle_s * HZ;
    if (ended_stable) close_plateau();
    // Litter poured in with the lid off (this event or an earlier one): the
    // box can then stand at its lid-on level with the lid still off.
    if (cfg_.lid_g > 0.0f && lid_off_ && ended_inert) {
      int ls = ladder_start();
      float lvl = pl_count_ > 0 ? level(pl_count_ - 1) : with_lid(win_.mean(), lid_off_);
      float from = ls < pl_count_ ? std::min(ev_base(), level(ls)) : ev_base();
      if (lvl - from >= cfg_.top_up_min_g && monotonic_from(ls)) lid_off_filled_ = true;
    }
    // The lid is "off" but the box stands where it did with the lid on, and
    // nothing went in: the lid never came off (or went back on unseen). With
    // it off the box sits a lid's weight lower, and nothing a cat does lifts
    // it back. Without this a missed lid-on stays wrong until the next
    // cleaning (8 Nov: off from 11:48 on with the box back at its level).
    if (cfg_.lid_g > 0.0f && lid_off_ && !absent_ && !deep_clean_ && !std::isnan(lid_ref_g_)) {
      float actual = pl_count_ > 0 ? pl_[pl_count_ - 1].mean_g : win_.mean();
      // Unless it was topped up while off (scooped, then refilled, lid still
      // off: that would read as the lid back and 485 g lost).
      if (!lid_off_filled_ && actual >= lid_ref_g_ - (cfg_.lid_g - cfg_.lid_tol_g) &&
          actual <= lid_ref_g_ + cfg_.top_up_min_g) {
        lid_off_ = pl_lid_off_ = ev_lid_off_ = lid_on_pending_ = false;
        for (int i = 0; i < pl_count_; i++) pl_[i].lid_off = false;
        lid_healed_ = true;
      }
    }
    BoxEvent &r = result_;
    r.kind = BoxEventKind::NONE;
    r.level_g = pl_count_ > 0 ? level(pl_count_ - 1) : with_lid(win_.mean(), lid_off_);
    // Ended the moment the lid went back on (take_lid_back): the box has held
    // still at its new level for only a second or so, too short to close as
    // a plateau, and the last closed one is from before the lid went on and
    // the last scoopful came out (8 Nov 23:29: 105 g low). Take the new level.
    if (!ended_stable && pl_n_ >= LID_HOLD && lid_level_ok_ && !pl_cat_ && !lid_off_)
      r.level_g = with_lid(static_cast<float>(pl_mean_), pl_lid_off_);
    r.litter_added_g = 0.0f;
    r.box_measured_g = box_measured_g_;
    r.zero_error_g = zero_error_g_;
    r.zero_valid = zero_valid_;
    r.absent = absent_;
    r.scooped = false;
    r.longest_still_s = inert_max_ / static_cast<float>(HZ);
    r.lid_off = lid_off_;
    r.lid_healed = lid_healed_;
    r.lid_delta_g = ev_lid_off_ == lid_off_ ? 0.0f : lid_off_ ? -cfg_.lid_g : lid_on_step_g_;
    float base = ev_base();
    float delta = r.level_g - base;

    if (absent_) {
      // Still off or in the air: nothing to conclude until it comes back.
      r.kind = state_ == BoxState::LIFTED ? BoxEventKind::LIFTED : BoxEventKind::NONE;
      return r;
    }
    if (deep_clean_) {
      r.kind = BoxEventKind::DEEP_CLEAN;
      // The box came back empty, so whatever is in it now went in since.
      if (r.level_g > box_measured_g_ + cfg_.top_up_min_g)
        r.litter_added_g = r.level_g - box_measured_g_;
      return r;
    }
    // The whole monitor went up and came back down: a few grams of settling
    // is not a scoop, the box was never opened.
    if (saw_lifted_ && std::abs(delta) < cfg_.return_tol_g) {
      r.kind = BoxEventKind::LIFTED;
      return r;
    }
    // The box went off and came back with nothing taken out or put in.
    if ((saw_off_ || began_absent_) && delta > -cfg_.scoop_min_g &&
        delta < cfg_.return_tol_g) {
      r.kind = BoxEventKind::LIFTED;
      return r;
    }
    // A rise that went inert is litter whatever it weighs: a bag can match a
    // cat to the gram, but no cat holds still this long. Measured from the
    // lowest point of the event rather than its start, so scooping before
    // the pour is not subtracted from the bag (it is reported alongside).
    int ls = ladder_start();
    float from = ls < pl_count_ ? std::min(base, level(ls)) : base;
    if (ended_inert && r.level_g - from >= cfg_.top_up_min_g && monotonic_from(ls)) {
      r.kind = BoxEventKind::TOP_UP;
      r.litter_added_g = r.level_g - from;
      r.scooped = base - from >= cfg_.scoop_min_g;
      return r;
    }
    // A lift-and-return needs no minimum duration: the absence already says
    // the box was opened, so a drop is a scoop however quick the hands were.
    // So does taking the lid off.
    if (!cat_event && delta <= -cfg_.scoop_min_g &&
        (duration_s >= cfg_.scoop_min_s || began_absent_ || saw_off_ || lid_opened_)) {
      r.kind = BoxEventKind::SCOOP;
      return r;
    }
    return r;
  }

  int plateau_count() const { return pl_count_; }
  const Plateau &plateau(int i) const { return pl_[i]; }

 private:
  static constexpr int HZ = 10;
  static constexpr int WINDOW = 10;
  /** Same bar as the analyzer's stability: sqrt(250). */
  static constexpr float PLATEAU_SIGMA = 15.811388f;
  /** Noise probe: spread histogram 0..5 g, up to 20 slices (10 min at 30 s). */
  static constexpr int PROBE_BINS = 100;
  static constexpr float PROBE_BIN_G = 0.05f;
  static constexpr int PROBE_MAX_SLICES = 20;
  static constexpr int MIN_PLATEAU = 2 * HZ;
  static constexpr int HOLD = 3 * HZ;
  /**
   * Successive plateau means jitter by a few grams (cell creep after a kilo
   * lands, a hand off the rim). A drop bigger than this is mass leaving.
   */
  static constexpr float LEVEL_TOL = 50.0f;
  /**
   * Coming off, the first plateau may already have a scoopful out, so up to
   * this much more than lid_g + lid_tol_g is accepted as the lid.
   */
  static constexpr float LID_SCOOP_G = 150.0f;
  /**
   * Straight down from the lid-on level in one step, the lid plus at most
   * this: a first scoopful can be out before the box first holds still (the
   * biggest single scoop on record took 236 g), but a 1092 g weight lifted
   * off the lid is not the lid (8 Nov).
   */
  static constexpr float LID_FIRST_SCOOP_G = 400.0f;
  /** A lid lifted off for a moment holds still for only a second or two. */
  static constexpr int LID_HOLD = HZ;
  /**
   * Seating a lid presses 1.9-8.3 kg above the level it leaves (17 of 17
   * recorded); a cat landing overshoots its own weight by 1.3 kg at most (54).
   */
  static constexpr float LID_PRESS_G = 1500.0f;
  /** A lid going on that also had a scoop come out still adds at least this. */
  static constexpr float LID_ON_MIN_G = 200.0f;
  /** ...and the box goes dead still within this many samples of the new level. */
  static constexpr int LID_STILL_WITHIN = 9 * HZ;
  /** A level this far above the lid-on box has a cat on it. */
  static constexpr float CAT_G = 2000.0f;
  /**
   * Weighing this much more than the box for over CAT_PRESS_N samples between
   * two levels: a cat, not a hand (leaning on the box while scooping has held
   * it there for 4 s).
   */
  static constexpr float CAT_PRESS_G = 2500.0f;
  static constexpr int CAT_PRESS_N = 8 * HZ;
  /** A level the lid is judged from holds within this (1 s spread). */
  static constexpr float LID_LEVEL_SIGMA = 4.0f;
  /** A lid set down goes from one level to the next within this; a pour takes longer. */
  static constexpr int LID_QUICK = 25;
  /** A cat gets on within this; a pour takes longer. */
  static constexpr int CAT_JUMP = 30;

  BoxConfig cfg_{};
  Ring win_{WINDOW};
  int n_ = 0;
  BoxState state_ = BoxState::NORMAL;
  bool absent_ = false;
  int hold_ = 0;
  int inert_len_ = 0;
  int inert_max_ = 0;
  int probe_left_ = 0;
  int probe_total_ = 0;
  bool probe_finished_ = false;
  uint16_t probe_hist_[PROBE_BINS] = {};
  int probe_slices_ = 0;
  int probe_slice_n_ = 0;
  float probe_slice_max_ = 0.0f;
  float probe_maxes_[PROBE_MAX_SLICES] = {};
  // open plateau (Welford, double: sums of 1e4 g over 1e4 samples lose float bits)
  int pl_start_ = 0;
  int pl_n_ = 0;
  double pl_mean_ = 0.0;
  double pl_m2_ = 0.0;
  // per event
  float base_g_ = 0.0f;
  int ev_start_ = 0;
  Plateau pl_[BM_MAX_PLATEAUS];
  int pl_count_ = 0;
  bool pl_overflow_ = false;
  bool saw_lifted_ = false;
  bool saw_off_ = false;
  bool began_absent_ = false;
  bool deep_clean_ = false;
  int empty_pl_ = -1;
  float box_measured_g_ = 0.0f;
  float zero_error_g_ = 0.0f;
  bool zero_valid_ = false;
  // lid (cfg_.lid_g > 0): lid_off_ and last_level_g_ carry over between events
  bool lid_off_ = false;
  bool pl_lid_off_ = false;   // lid state of the open plateau
  float last_level_g_ = NAN;  // last level held LID_HOLD with the box on the board
  float lid_ref_g_ = NAN;     // where the box stands with the lid on (the level it came off from)
  bool ev_lid_off_ = false;   // lid state when the event began
  bool lid_opened_ = false;   // the lid came off or went on this event
  bool lid_off_filled_ = false;  // litter was topped up since the lid came off
  int lid_off_n_ = -1;        // sample the lid was last seen coming off
  float lid_on_step_g_ = 0.0f;  // what the lid added going back on
  float trans_max_ = -INFINITY;  // hardest press since the last level
  int trans_heavy_ = 0;          // longest run weighing like a cat since the last level
  int heavy_run_ = 0;
  bool lid_level_ok_ = false;    // the open plateau held still enough and the lid was judged on it
  bool lid_on_pending_ = false;  // pressed on, waiting for the box to go still
  float lid_on_pending_d_ = 0.0f;
  int lid_on_deadline_ = 0;
  bool hold_broken_ = false;     // a cat got on while the event was held open
  bool lid_back_ = false;        // the lid just went back on: the cleaning can end
  bool lid_healed_ = false;      // finalize found the lid on after all
  bool ref_stale_ = false;       // the box was away: re-learn the levels when it is back
  int last_level_end_n_ = 0;     // sample the last level ended
  bool pl_cat_ = false;          // the open plateau has a cat on it
  float box_level_g_ = NAN;      // the latest still level with no cat on
  bool prev_cat_ = false;        // the last level judged had a cat on it
  BoxEvent result_{};

  /** Below anything the box can weigh: it is off the board (or the board is in the air). */
  bool below_box(float g) const {
    return cfg_.box_g > 0.0f ? g < 0.5f * cfg_.box_g : g < cfg_.off_tol_g;
  }

  /** A level as if the lid were on. */
  float with_lid(float g, bool lid_off) const { return lid_off ? g + cfg_.lid_g : g; }
  float level(int i) const { return with_lid(pl_[i].mean_g, pl_[i].lid_off); }
  /** What the event is measured against, lid on. */
  float ev_base() const { return with_lid(base_g_, ev_lid_off_); }

  bool empty_without_lid(float g) const {
    return cfg_.lid_g > 0.0f && std::abs(g + cfg_.lid_g - cfg_.box_g) < cfg_.empty_tol_g;
  }

  void set_lid_on(float d) {
    // Off and back on within this event, nothing added: the cleaning is over.
    lid_back_ = lid_off_n_ >= ev_start_ &&
                with_lid(static_cast<float>(pl_mean_), false) - ev_base() < cfg_.top_up_min_g;
    lid_off_ = pl_lid_off_ = false;
    lid_opened_ = true;
    lid_off_filled_ = false;
    // What it actually added back, if that looks like the lid alone.
    lid_on_step_g_ = std::abs(d - cfg_.lid_g) <= cfg_.lid_tol_g ? d : cfg_.lid_g;
    lid_ref_g_ = static_cast<float>(pl_mean_);
  }

  /** Once per plateau, on its first still second: did the lid just come off or go back? */
  void track_lid(float g) {
    pl_lid_off_ = lid_off_;
    int trans_len = pl_start_ - last_level_end_n_;
    if (ref_stale_ && !below_box(g)) {
      // Back from being away: start over from here, lid as it was.
      ref_stale_ = false;
      lid_ref_g_ = with_lid(g, lid_off_);
      last_level_g_ = box_level_g_ = g;
      trans_max_ = g;
      trans_heavy_ = 0;
      last_level_end_n_ = n_;
      return;
    }
    float press = trans_max_ - g;
    // With the lid on, anything well above the box is a cat. With it off,
    // litter can be poured in: a cat jumps on (2 kg or more in a few
    // seconds) and stays a cat until the box is back near where it was;
    // a pour climbs for longer and then is just the box, heavier.
    float rise = std::isnan(last_level_g_) ? 0.0f : g - last_level_g_;
    bool cat = trans_heavy_ > CAT_PRESS_N ||
               (!lid_off_ ? !std::isnan(lid_ref_g_) && g > lid_ref_g_ + CAT_G
                          : rise > CAT_G && (trans_len <= CAT_JUMP || prev_cat_));
    pl_cat_ = prev_cat_ = cat;
    heavy_run_ = 0;
    if (!cat) box_level_g_ = g;
    trans_max_ = g;
    trans_heavy_ = 0;
    // A cat got in with the lid off: from here the event ends as a visit does.
    if (cat && lid_off_ && lid_off_n_ >= ev_start_) hold_broken_ = true;
    if (cfg_.lid_g <= 0.0f || below_box(g) || std::isnan(last_level_g_) || cat) return;
    float d = g - last_level_g_;
    float tol = cfg_.lid_tol_g;
    if (!lid_off_) {
      bool step = d <= -(cfg_.lid_g - tol) && d >= -(cfg_.lid_g + tol + LID_SCOOP_G);
      // The first level may still have the lid resting on the box or a hand
      // on it; once one holds a lid's weight below the start, it is off.
      // Straight down from the lid-on level in one step, only a lid's worth
      // (plus a first scoopful, LID_FIRST_SCOOP_G) counts: more than that at once is something
      // else coming off, such as a weight set on the lid and lifted away
      // again (8 Nov: 1092 g off read as lid off + scoop). Via a level in
      // between (a hand still on the box, 21 Oct) it can be any amount.
      bool from_ref = !std::isnan(lid_ref_g_) && std::abs(last_level_g_ - lid_ref_g_) <= LEVEL_TOL;
      bool below = !std::isnan(lid_ref_g_) && g <= lid_ref_g_ - (cfg_.lid_g - tol) &&
                   (!from_ref || g >= lid_ref_g_ - (cfg_.lid_g + tol + LID_FIRST_SCOOP_G));
      if (from_ref && g < lid_ref_g_ - (cfg_.lid_g + tol + LID_FIRST_SCOOP_G)) {
        // Something else came off with the lid still on: it stands here now.
        lid_ref_g_ = g;
      } else if (step || below) {
        lid_off_ = pl_lid_off_ = lid_opened_ = true;
        lid_off_n_ = n_;
        lid_off_filled_ = false;
        if (std::isnan(lid_ref_g_)) lid_ref_g_ = last_level_g_;
      }
    } else if ((d >= LID_ON_MIN_G && press >= LID_PRESS_G) ||
               // Set down gently: no press, but a lid-sized step in a moment
               // (a pour takes longer), and the box then goes dead still.
               (std::abs(d - cfg_.lid_g) <= tol && trans_len <= LID_QUICK)) {
      // Pressed on and about its weight: the lid. Otherwise (a scoop went
      // with it, or no press): the lid once the box goes still.
      if (std::abs(d - cfg_.lid_g) <= tol && press >= LID_PRESS_G) {
        set_lid_on(d);
      } else {
        lid_on_pending_ = true;
        lid_on_pending_d_ = d;
        lid_on_deadline_ = n_ + LID_STILL_WITHIN;
      }
    }
  }

  /** Lowest plateau since the box was last away: where a pour starts from. */
  int ladder_start() const {
    int s = 0;
    for (int i = 0; i < pl_count_; i++)
      if (below_box(pl_[i].mean_g)) s = i + 1;
    int lo = s;
    for (int i = s + 1; i < pl_count_; i++)
      if (level(i) < level(lo)) lo = i;
    return lo;
  }

  void close_plateau() {
    if (pl_n_ >= MIN_PLATEAU && pl_start_ >= ev_start_) {
      float sigma = pl_n_ > 1 ? sqrtf(static_cast<float>(pl_m2_ / (pl_n_ - 1))) : 0.0f;
      if (pl_count_ < BM_MAX_PLATEAUS)
        pl_[pl_count_++] = {pl_start_, pl_start_ + pl_n_, static_cast<float>(pl_mean_), sigma,
                            pl_lid_off_};
      else
        pl_overflow_ = true;
    }
    // A cat sitting still is not a level the lid moves from.
    if (pl_n_ >= LID_HOLD && lid_level_ok_ && !pl_cat_ && !below_box(static_cast<float>(pl_mean_))) {
      last_level_g_ = static_cast<float>(pl_mean_);
      last_level_end_n_ = n_;
    }
    pl_n_ = 0;
    pl_mean_ = 0.0;
    pl_m2_ = 0.0;
    lid_level_ok_ = false;
    pl_cat_ = false;
  }

  float current_level() const {
    return pl_n_ >= MIN_PLATEAU ? static_cast<float>(pl_mean_) : win_.mean();
  }

  /** Plateaus from `from` on never drop by more than the jitter tolerance. */
  bool monotonic_from(int from) const {
    if (pl_overflow_) return false;
    for (int i = from + 1; i < pl_count_; i++)
      if (level(i) < level(i - 1) - LEVEL_TOL) return false;
    return true;
  }
};

// Singleton accessors for use across ESPHome lambdas
inline StateAnalyzer &get_analyzer() {
  static StateAnalyzer instance;
  return instance;
}
inline WeightBuffer &get_weight_buf() {
  static WeightBuffer instance;
  return instance;
}
inline BoxMonitor &get_box_monitor() {
  static BoxMonitor instance;
  return instance;
}
inline int &get_sample_idx() {
  static int idx = 0;
  return idx;
}

inline const char *box_state_str(BoxState s) {
  switch (s) {
    case BoxState::NORMAL: return "normal";
    case BoxState::LIFTED: return "lifted";
    case BoxState::OFF:    return "off";
    case BoxState::EMPTY:  return "empty";
  }
  return "unknown";
}

inline const char *box_event_str(BoxEventKind k) {
  switch (k) {
    case BoxEventKind::NONE:       return "none";
    case BoxEventKind::LIFTED:     return "lifted";
    case BoxEventKind::DEEP_CLEAN: return "deep_clean";
    case BoxEventKind::TOP_UP:     return "top_up";
    case BoxEventKind::SCOOP:      return "scoop";
  }
  return "none";
}

inline const char *analyzer_state_str(AnalyzerState s) {
  switch (s) {
    case AnalyzerState::EMPTY:       return "empty";
    case AnalyzerState::ENTERING:    return "entering";
    case AnalyzerState::OCCUPIED:    return "occupied";
    case AnalyzerState::ELIMINATING: return "eliminating";
    case AnalyzerState::GAP:         return "gap";
  }
  return "unknown";
}

inline const char *elimination_type_str(EliminationType t) {
  switch (t) {
    case EliminationType::NO_ELIMINATION: return "no_elimination";
    case EliminationType::URINATION:      return "urination";
    case EliminationType::DEFECATION:     return "defecation";
    case EliminationType::BOTH:           return "both";
    case EliminationType::UNKNOWN:        return "unknown";
  }
  return "unknown";
}
