use std::collections::VecDeque;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpikeCoincidence {
    pub pre_syn_nid: usize,
    pub syn_idx: usize,
    pub t_pre_minus_post: i64,
}

#[derive(Debug, Default, Clone)]
pub struct SpikeCoincidenceDetector {
    recent_pre_syn_spikes: VecDeque<PreSynSpike>, // TODO: SmallVec?
}

impl SpikeCoincidenceDetector {
    pub fn on_psp(
        &mut self,
        t: usize,
        last_post_syn_spike_t: Option<usize>,
        pre_syn_nid: usize,
        syn_idx: usize,
        t_cutoff: usize,
    ) -> Option<SpikeCoincidence> {
        self.discard_stale(t, t_cutoff);

        self.recent_pre_syn_spikes.push_back(PreSynSpike {
            nid: pre_syn_nid,
            syn_idx,
            t_transmission: t,
        });

        match last_post_syn_spike_t {
            Some(last_post_syn_spike_t) if last_post_syn_spike_t + t_cutoff >= t => {
                let t_pre_minus_post = (t - last_post_syn_spike_t) as i64;
                Some(SpikeCoincidence {
                    pre_syn_nid,
                    syn_idx,
                    t_pre_minus_post,
                })
            }
            _ => None,
        }
    }

    pub fn on_post_syn_spike<'a>(
        &'a mut self,
        t: usize,
        t_cutoff: usize,
    ) -> impl Iterator<Item = SpikeCoincidence> + 'a {
        self.discard_stale(t, t_cutoff);

        self.recent_pre_syn_spikes
            .drain(..)
            .map(move |pre_syn_spike| {
                let t_pre_minus_post = -((t - pre_syn_spike.t_transmission) as i64);
                SpikeCoincidence {
                    pre_syn_nid: pre_syn_spike.nid,
                    syn_idx: pre_syn_spike.syn_idx,
                    t_pre_minus_post,
                }
            })
    }

    fn discard_stale(&mut self, t: usize, t_cutoff: usize) {
        while let Some(oldest_spike) = self.recent_pre_syn_spikes.front() {
            if oldest_spike.t_transmission + t_cutoff < t {
                self.recent_pre_syn_spikes.pop_front();
            } else {
                break;
            }
        }
    }
}

#[derive(Debug, Clone)]
struct PreSynSpike {
    nid: usize,
    syn_idx: usize,
    t_transmission: usize,
}

#[cfg(test)]
mod tests {
    use crate::spike_coincidence::SpikeCoincidence;

    use super::SpikeCoincidenceDetector;

    const PRE_SYN_NID_0: usize = 10;
    const PRE_SYN_NID_1: usize = 11;
    const SYN_IDX: usize = 20;
    const T_CUTOFF: usize = 10;

    #[test]
    fn post_syn_spike_after_cutoff() {
        let mut sut = SpikeCoincidenceDetector::default();

        assert!(sut
            .on_psp(1, None, PRE_SYN_NID_0, SYN_IDX, T_CUTOFF)
            .is_none());

        assert!(sut
            .on_psp(2, None, PRE_SYN_NID_1, SYN_IDX, T_CUTOFF)
            .is_none());

        let mut coincidences = sut.on_post_syn_spike(13, T_CUTOFF);
        assert!(coincidences.next().is_none());
    }

    #[test]
    fn post_syn_spike_before_cutoff() {
        let mut sut = SpikeCoincidenceDetector::default();

        assert!(sut
            .on_psp(1, None, PRE_SYN_NID_0, SYN_IDX, T_CUTOFF)
            .is_none());

        assert!(sut
            .on_psp(2, None, PRE_SYN_NID_1, SYN_IDX, T_CUTOFF)
            .is_none());

        let coincidences = sut.on_post_syn_spike(5, T_CUTOFF);
        assert_eq!(
            coincidences.collect::<Vec<SpikeCoincidence>>(),
            vec![
                SpikeCoincidence {
                    pre_syn_nid: PRE_SYN_NID_0,
                    syn_idx: SYN_IDX,
                    t_pre_minus_post: -4
                },
                SpikeCoincidence {
                    pre_syn_nid: PRE_SYN_NID_1,
                    syn_idx: SYN_IDX,
                    t_pre_minus_post: -3
                }
            ]
        );

        // check that pre-synaptic spikes are not double accounted
        let mut coincidences = sut.on_post_syn_spike(6, T_CUTOFF);
        assert!(coincidences.next().is_none());
    }

    #[test]
    fn post_syn_spike_subset_cutoff() {
        let mut sut = SpikeCoincidenceDetector::default();

        assert!(sut
            .on_psp(1, None, PRE_SYN_NID_0, SYN_IDX, T_CUTOFF)
            .is_none());

        assert!(sut
            .on_psp(2, None, PRE_SYN_NID_1, SYN_IDX, T_CUTOFF)
            .is_none());

        let coincidences = sut.on_post_syn_spike(12, T_CUTOFF);
        assert_eq!(
            coincidences.collect::<Vec<SpikeCoincidence>>(),
            vec![SpikeCoincidence {
                pre_syn_nid: PRE_SYN_NID_1,
                syn_idx: SYN_IDX,
                t_pre_minus_post: -10
            }]
        );

        // check that pre-synaptic spikes are not double accounted
        let mut coincidences = sut.on_post_syn_spike(6, T_CUTOFF);
        assert!(coincidences.next().is_none());
    }

    #[test]
    fn pre_syn_spike_after_cutoff() {
        let mut sut = SpikeCoincidenceDetector::default();

        assert!(sut.on_post_syn_spike(1, T_CUTOFF).next().is_none());

        assert!(sut
            .on_psp(12, Some(1), PRE_SYN_NID_0, SYN_IDX, T_CUTOFF)
            .is_none());
    }

    #[test]
    fn pre_syn_spike_before_cutoff() {
        let mut sut = SpikeCoincidenceDetector::default();

        assert!(sut.on_post_syn_spike(1, T_CUTOFF).next().is_none());

        assert_eq!(
            sut.on_psp(6, Some(1), PRE_SYN_NID_0, SYN_IDX, T_CUTOFF),
            Some(SpikeCoincidence {
                pre_syn_nid: PRE_SYN_NID_0,
                syn_idx: SYN_IDX,
                t_pre_minus_post: 5
            })
        );

        assert_eq!(
            sut.on_psp(6, Some(1), PRE_SYN_NID_1, SYN_IDX, T_CUTOFF),
            Some(SpikeCoincidence {
                pre_syn_nid: PRE_SYN_NID_1,
                syn_idx: SYN_IDX,
                t_pre_minus_post: 5
            })
        );

        assert_eq!(
            sut.on_psp(7, Some(1), PRE_SYN_NID_0, SYN_IDX, T_CUTOFF),
            Some(SpikeCoincidence {
                pre_syn_nid: PRE_SYN_NID_0,
                syn_idx: SYN_IDX,
                t_pre_minus_post: 6
            })
        );
    }
}
