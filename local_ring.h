#pragma once
// Local build: no visit journal on flash (upstream's MQTT feature isn't used
// here), so the analyzer keeps the whole visit in RAM, as it did before
// upstream #12. Without this, a visit longer than ~7 minutes gets no verdict.
#define SA_RING_SAMPLES SA_MAX_SAMPLES
