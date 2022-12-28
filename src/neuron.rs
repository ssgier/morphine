use crate::{
    params::NeuronParams,
    spike_coincidence::{SpikeCoincidence, SpikeCoincidenceDetector},
    util::get_decay_factor,
};

#[derive(Debug, Clone)]
pub struct Neuron {
    last_t: usize,
    last_spike_t: Option<usize>,
    last_voltage: f32,
    last_threshold: f32,
    spike_coincidence_detector: SpikeCoincidenceDetector,
}

#[derive(Debug, Clone)]
pub struct ApplyPspResult {
    pub might_spike: bool,
    pub spike_coincidence: Option<SpikeCoincidence>,
}

#[derive(Debug, Clone)]
pub struct Spike<T: Iterator<Item = SpikeCoincidence>>(pub T);

impl Neuron {
    pub fn new() -> Self {
        Self {
            last_t: 0,
            last_spike_t: None,
            last_voltage: 0.0,
            last_threshold: 1.0,
            spike_coincidence_detector: SpikeCoincidenceDetector::default(),
        }
    }

    pub fn get_last_spike_t(&self) -> Option<usize> {
        self.last_spike_t
    }

    pub fn apply_psp(
        &mut self,
        t: usize,
        psp: f32,
        pre_syn_nid: usize,
        syn_idx: usize,
        neuron_params: &NeuronParams,
    ) -> ApplyPspResult {
        let spike_coincidence = self.spike_coincidence_detector.on_psp(
            t,
            self.last_spike_t,
            pre_syn_nid,
            syn_idx,
            neuron_params.t_cutoff_coincidence,
        );

        let might_spike = if !self.is_refractory(t) {
            let decay_factor_voltage = get_decay_factor(t, self.last_t, neuron_params.tau_membrane);
            let decay_factor_threshold =
                get_decay_factor(t, self.last_t, neuron_params.tau_threshold);

            // retroactively adjust last voltage, because in current cycle there might still be excitatory spikes to be processed
            if self.last_voltage < neuron_params.voltage_floor && self.last_t < t {
                self.last_voltage = neuron_params.voltage_floor;
            }

            self.last_t = t;

            self.last_voltage *= decay_factor_voltage;
            self.last_voltage += psp;

            self.last_threshold = 1.0 + (self.last_threshold - 1.0) * decay_factor_threshold;

            self.last_voltage >= self.last_threshold
        } else {
            false
        };

        ApplyPspResult {
            might_spike,
            spike_coincidence,
        }
    }

    pub fn check_spike(
        &mut self,
        t: usize,
        neuron_params: &NeuronParams,
    ) -> Option<Spike<impl Iterator<Item = SpikeCoincidence> + '_>> {
        if self.last_voltage >= self.last_threshold {
            Some(self.spike(t, neuron_params))
        } else {
            None
        }
    }

    pub fn spike(
        &mut self,
        t: usize,
        neuron_params: &NeuronParams,
    ) -> Spike<impl Iterator<Item = SpikeCoincidence> + '_> {
        self.last_spike_t = Some(t);
        self.last_t = t + neuron_params.refractory_period as usize;
        self.last_voltage = neuron_params.reset_voltage;
        self.last_threshold = neuron_params.adaptation_threshold;

        Spike(
            self.spike_coincidence_detector
                .on_post_syn_spike(t, neuron_params.t_cutoff_coincidence),
        )
    }

    fn is_refractory(&self, t: usize) -> bool {
        self.last_t > t
    }

    pub fn get_voltage(&self, t: usize, neuron_params: &NeuronParams) -> f32 {
        let adjusted_last_voltage = self.last_voltage.max(neuron_params.voltage_floor);

        if self.last_t >= t {
            adjusted_last_voltage
        } else {
            let decay_factor_voltage = get_decay_factor(t, self.last_t, neuron_params.tau_membrane);
            adjusted_last_voltage * decay_factor_voltage
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use float_cmp::assert_approx_eq;

    fn params() -> NeuronParams {
        NeuronParams {
            tau_membrane: 10.0,
            refractory_period: 10,
            reset_voltage: 0.0,
            t_cutoff_coincidence: 20,
            adaptation_threshold: 1.0,
            tau_threshold: 50.0,
            voltage_floor: 0.0,
        }
    }

    #[test]
    fn leak() {
        let params = params();
        let mut sut = Neuron::new();

        let psp_result = sut.apply_psp(4, 0.5, 0, 0, &params);
        assert!(!psp_result.might_spike);
        assert_eq!(sut.last_t, 4);
        assert_eq!(sut.last_spike_t, None);
        assert_approx_eq!(f32, sut.last_voltage, 0.5);

        let psp_result = sut.apply_psp(14, 0.5, 0, 0, &params);
        assert!(!psp_result.might_spike);
        assert_eq!(sut.last_t, 14);
        assert_eq!(sut.last_spike_t, None);
        assert_approx_eq!(f32, sut.last_voltage, 0.5 * (-1.0f32).exp() + 0.5);
    }

    #[test]
    fn spike_after_leakage() {
        let params = params();
        let mut sut = Neuron::new();

        sut.apply_psp(4, 0.5, 0, 0, &params);

        let psp_result = sut.apply_psp(14, 0.9, 0, 0, &params);

        assert!(psp_result.might_spike);
        assert_eq!(sut.last_t, 14);
        assert_eq!(sut.last_spike_t, None);
        assert_approx_eq!(f32, sut.last_voltage, 0.5 * (-1.0f32).exp() + 0.9);

        let spike = sut.check_spike(14, &params);
        assert!(spike.is_some());
        drop(spike);
        assert_eq!(sut.last_t, 14 + params.refractory_period as usize);
        assert_eq!(sut.last_spike_t, Some(14));
        assert_approx_eq!(f32, sut.last_voltage, params.reset_voltage);
    }

    #[test]
    fn coinciding_psps_cause_spike() {
        let params = params();
        let mut sut = Neuron::new();

        sut.apply_psp(4, 0.5, 0, 0, &params);

        let psp_result = sut.apply_psp(4, 0.5, 0, 0, &params);

        assert!(psp_result.might_spike);
        assert_eq!(sut.last_t, 4);
        assert_eq!(sut.last_spike_t, None);
        assert_approx_eq!(f32, sut.last_voltage, 1.0);

        let spike = sut.check_spike(4, &params);
        assert!(spike.is_some());
        drop(spike);
        assert_eq!(sut.last_t, 4 + params.refractory_period as usize);
        assert_eq!(sut.last_spike_t, Some(4));
        assert_approx_eq!(f32, sut.last_voltage, params.reset_voltage);
    }

    #[test]
    fn check_spike_no_duplicate() {
        let params = params();
        let mut sut = Neuron::new();
        sut.apply_psp(1, 1.0, 0, 0, &params);

        assert!(sut.check_spike(1, &params).is_some());
        assert!(sut.check_spike(1, &params).is_none());
    }

    #[test]
    fn spike_coincidence_pre_then_post() {
        let params = params();
        let mut sut = Neuron::new();
        assert!(sut
            .apply_psp(4, 0.5, 0, 1, &params)
            .spike_coincidence
            .is_none());

        let spike = sut.spike(6, &params);

        assert_eq!(
            spike.0.collect::<Vec<_>>(),
            vec![SpikeCoincidence {
                pre_syn_nid: 0,
                syn_idx: 1,
                t_pre_minus_post: -2
            }]
        )
    }

    #[test]
    fn spike_coincidence_post_then_pre() {
        let params = params();
        let mut sut = Neuron::new();

        let mut spike = sut.spike(1, &params);
        assert!(spike.0.next().is_none());
        drop(spike);

        let psp_result = sut.apply_psp(5, 0.1, 0, 1, &params);
        assert_eq!(
            psp_result.spike_coincidence,
            Some(SpikeCoincidence {
                pre_syn_nid: 0,
                syn_idx: 1,
                t_pre_minus_post: 4
            })
        );
    }

    #[test]
    fn single_psp_spike_coincidence() {
        let params = params();
        let mut sut = Neuron::new();

        assert!(sut.apply_psp(1, 1.0, 0, 1, &params).might_spike);

        let spike = sut.check_spike(1, &params);

        assert_eq!(
            spike.unwrap().0.collect::<Vec<_>>(),
            vec![SpikeCoincidence {
                pre_syn_nid: 0,
                syn_idx: 1,
                t_pre_minus_post: 0
            }]
        );
    }

    #[test]
    fn threshold_adaptation() {
        let mut params = params();
        params.adaptation_threshold = 2.0;
        params.tau_threshold = 100.0;

        let mut sut = Neuron::new();

        assert_approx_eq!(f32, sut.last_threshold, 1.0);
        sut.spike(0, &params);
        assert_approx_eq!(f32, sut.last_threshold, 2.0);

        assert!(!sut.apply_psp(20, 1.0, 0, 0, &params).might_spike);
        let expected_threshold = 1.0 + (-10.0 / 100f32).exp();
        assert_approx_eq!(f32, sut.last_threshold, expected_threshold);

        assert!(
            sut.apply_psp(20, expected_threshold - 1.0, 0, 0, &params)
                .might_spike
        )
    }

    #[test]
    fn voltage_floor() {
        let mut params = params();
        params.voltage_floor = -0.5;
        params.tau_membrane = 1000.0;
        let mut sut = Neuron::new();

        sut.apply_psp(1, -0.4, 0, 0, &params);
        assert_approx_eq!(f32, sut.get_voltage(1, &params), -0.4);
        sut.apply_psp(1, -0.4, 0, 0, &params);

        // epsp too small to lift above floor
        sut.apply_psp(1, 0.1, 0, 0, &params);

        // raw voltage is not yet floor-adjusted
        assert_approx_eq!(f32, sut.last_voltage, -0.7);
        assert_approx_eq!(f32, sut.get_voltage(1, &params), -0.5);

        sut.apply_psp(2, 0.0, 0, 0, &params);

        // now raw voltage should have been retroactively adjusted
        let expected_voltage = -0.5 * (-1.0 / 1000.0f32).exp();
        assert_approx_eq!(f32, sut.last_voltage, expected_voltage);
        assert_approx_eq!(f32, sut.get_voltage(1, &params), expected_voltage);
    }

    #[test]
    fn ipsp_prevents_spike() {
        let params = params();
        let mut sut = Neuron::new();

        sut.apply_psp(4, 0.5, 0, 0, &params);
        sut.apply_psp(4, 0.6, 0, 0, &params);
        sut.apply_psp(4, -0.2, 0, 0, &params);

        assert_eq!(sut.last_t, 4);
        assert_eq!(sut.last_spike_t, None);
        assert_approx_eq!(f32, sut.last_voltage, 0.9);
        assert!(sut.check_spike(4, &params).is_none());
    }
}
