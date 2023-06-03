use std::collections::VecDeque;

use crate::util::SynapseCoordinate;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpikeCoincidence {
    pub syn_coord: SynapseCoordinate,
    pub t_pre_minus_post: i64,
}

#[derive(Debug, Default, Clone)]
pub struct SpikeCoincidenceDetector {
    recent_pre_syn_spikes: VecDeque<PreSynSpike>,
}

impl SpikeCoincidenceDetector {
    pub fn on_psp(
        &mut self,
        t: usize,
        last_post_syn_spike_t: Option<usize>,
        syn_coord: &SynapseCoordinate,
        t_cutoff: usize,
    ) -> Option<SpikeCoincidence> {
        self.discard_stale(t, t_cutoff);

        self.recent_pre_syn_spikes.push_back(PreSynSpike {
            syn_coord: syn_coord.clone(),
            t_transmission: t,
        });

        match last_post_syn_spike_t {
            Some(last_post_syn_spike_t) if last_post_syn_spike_t + t_cutoff >= t => {
                let t_pre_minus_post = (t - last_post_syn_spike_t) as i64;
                Some(SpikeCoincidence {
                    syn_coord: syn_coord.clone(),
                    t_pre_minus_post,
                })
            }
            _ => None,
        }
    }

    pub fn on_post_syn_spike(
        &mut self,
        t: usize,
        t_cutoff: usize,
    ) -> impl Iterator<Item = SpikeCoincidence> + '_ {
        self.discard_stale(t, t_cutoff);

        self.recent_pre_syn_spikes
            .drain(..)
            .map(move |pre_syn_spike| {
                let t_pre_minus_post = -((t - pre_syn_spike.t_transmission) as i64);
                SpikeCoincidence {
                    syn_coord: pre_syn_spike.syn_coord,
                    t_pre_minus_post,
                }
            })
    }

    pub fn reset_ephemeral_state(&mut self) {
        self.clear_spike_coincidences();
    }

    pub fn clear_spike_coincidences(&mut self) {
        self.recent_pre_syn_spikes.clear();
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
    syn_coord: SynapseCoordinate,
    t_transmission: usize,
}

#[cfg(test)]
mod tests {
    use crate::{spike_coincidence::SpikeCoincidence, util::SynapseCoordinate};

    use super::SpikeCoincidenceDetector;

    const PRE_SYN_NID_0: usize = 10;
    const PRE_SYN_NID_1: usize = 11;
    const PRJ_IDX: usize = 2;
    const SYN_IDX: usize = 20;
    const T_CUTOFF: usize = 10;

    const SYN_COORD_0: SynapseCoordinate = SynapseCoordinate {
        pre_syn_nid: PRE_SYN_NID_0,
        projection_idx: PRJ_IDX,
        synapse_idx: SYN_IDX,
    };

    const SYN_COORD_1: SynapseCoordinate = SynapseCoordinate {
        pre_syn_nid: PRE_SYN_NID_1,
        projection_idx: PRJ_IDX,
        synapse_idx: SYN_IDX,
    };

    #[test]
    fn post_syn_spike_after_cutoff() {
        let mut sut = SpikeCoincidenceDetector::default();

        assert!(sut.on_psp(1, None, &SYN_COORD_0, T_CUTOFF).is_none());

        assert!(sut.on_psp(2, None, &SYN_COORD_1, T_CUTOFF).is_none());

        let mut coincidences = sut.on_post_syn_spike(13, T_CUTOFF);
        assert!(coincidences.next().is_none());
    }

    #[test]
    fn post_syn_spike_before_cutoff() {
        let mut sut = SpikeCoincidenceDetector::default();

        assert!(sut.on_psp(1, None, &SYN_COORD_0, T_CUTOFF).is_none());

        assert!(sut.on_psp(2, None, &SYN_COORD_1, T_CUTOFF).is_none());

        let coincidences = sut.on_post_syn_spike(5, T_CUTOFF);
        assert_eq!(
            coincidences.collect::<Vec<SpikeCoincidence>>(),
            vec![
                SpikeCoincidence {
                    syn_coord: SYN_COORD_0,
                    t_pre_minus_post: -4
                },
                SpikeCoincidence {
                    syn_coord: SYN_COORD_1,
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

        assert!(sut.on_psp(1, None, &SYN_COORD_0, T_CUTOFF).is_none());

        assert!(sut.on_psp(2, None, &SYN_COORD_1, T_CUTOFF).is_none());

        let coincidences = sut.on_post_syn_spike(12, T_CUTOFF);
        assert_eq!(
            coincidences.collect::<Vec<SpikeCoincidence>>(),
            vec![SpikeCoincidence {
                syn_coord: SYN_COORD_1,
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

        assert!(sut.on_psp(12, Some(1), &SYN_COORD_0, T_CUTOFF).is_none());
    }

    #[test]
    fn pre_syn_spike_before_cutoff() {
        let mut sut = SpikeCoincidenceDetector::default();

        assert!(sut.on_post_syn_spike(1, T_CUTOFF).next().is_none());

        assert_eq!(
            sut.on_psp(6, Some(1), &SYN_COORD_0, T_CUTOFF),
            Some(SpikeCoincidence {
                syn_coord: SYN_COORD_0,
                t_pre_minus_post: 5
            })
        );

        assert_eq!(
            sut.on_psp(6, Some(1), &SYN_COORD_1, T_CUTOFF),
            Some(SpikeCoincidence {
                syn_coord: SYN_COORD_1,
                t_pre_minus_post: 5
            })
        );

        assert_eq!(
            sut.on_psp(7, Some(1), &SYN_COORD_0, T_CUTOFF),
            Some(SpikeCoincidence {
                syn_coord: SYN_COORD_0,
                t_pre_minus_post: 6
            })
        );
    }
}
