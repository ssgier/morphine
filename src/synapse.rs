use crate::{params::SynapseParams, util::get_decay_factor};

pub struct Synapse {
    pub neuron_idx: usize,
    pub conduction_delay: u8,
    pub weight: f32,
    pub short_term_stdp_offset: f32,
    pub last_t: usize,
}

impl Synapse {
    pub fn new(neuron_idx: usize, conduction_delay: u8, initial_weight: f32) -> Self {
        Self {
            neuron_idx,
            conduction_delay,
            weight: initial_weight,
            short_term_stdp_offset: 0.0,
            last_t: 0,
        }
    }

    pub fn process_pre_syn_spike_get_psp(
        &mut self,
        t: usize,
        stp_value: f32,
        syn_params: &SynapseParams,
        short_term_stdp_tau: f32,
    ) -> f32 {
        let decay_factor = get_decay_factor(
            t + self.conduction_delay as usize,
            self.last_t,
            short_term_stdp_tau,
        );

        let short_term_stdp = self.short_term_stdp_offset * decay_factor;
        let adjusted_weight = (self.weight * stp_value + short_term_stdp).max(0.0);

        syn_params.weight_scale_factor * adjusted_weight
    }

    pub fn process_weight_change(&mut self, weight_change: f32, syn_params: &SynapseParams) {
        self.weight = (self.weight + weight_change)
            .max(0.0)
            .min(syn_params.max_weight);
    }

    pub fn process_short_term_stdp(&mut self, t: usize, stdp_value: f32, short_term_stdp_tau: f32) {
        self.update(t, short_term_stdp_tau);
        self.short_term_stdp_offset += stdp_value;
    }

    fn update(&mut self, t: usize, short_term_stdp_tau: f32) {
        let decay_factor = get_decay_factor(t, self.last_t, short_term_stdp_tau);
        self.short_term_stdp_offset *= decay_factor;
        self.last_t = t;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    const SYNAPSE_PARAMS: SynapseParams = SynapseParams {
        max_weight: 0.5,
        weight_scale_factor: 0.1,
    };

    const SHORT_TERM_STDP_TAU: f32 = 10.0;

    #[test]
    fn weight_change() {
        let mut sut = Synapse::new(0, 0, 0.4);
        assert_approx_eq!(f32, sut.weight, 0.4);
        sut.process_weight_change(-0.1, &SYNAPSE_PARAMS);
        assert_approx_eq!(f32, sut.weight, 0.3);
    }

    #[test]
    fn weight_floor() {
        let mut sut = Synapse::new(0, 0, 0.4);
        assert_approx_eq!(f32, sut.weight, 0.4);
        sut.process_weight_change(-0.5, &SYNAPSE_PARAMS);
        assert_approx_eq!(f32, sut.weight, 0.0);
    }

    #[test]
    fn weight_ceil() {
        let mut sut = Synapse::new(0, 0, 0.4);
        assert_approx_eq!(f32, sut.weight, 0.4);
        sut.process_weight_change(0.3, &SYNAPSE_PARAMS);
        assert_approx_eq!(f32, sut.weight, 0.5);
    }

    #[test]
    fn stp() {
        let mut sut = Synapse::new(0, 0, 0.4);
        let psp = sut.process_pre_syn_spike_get_psp(4, 2.0, &SYNAPSE_PARAMS, SHORT_TERM_STDP_TAU);
        assert_approx_eq!(f32, psp, 2.0 * 0.4 * SYNAPSE_PARAMS.weight_scale_factor);
    }

    #[test]
    fn short_term_stdp_decay_and_another() {
        let mut sut = Synapse::new(0, 0, 0.4);

        // first short term stdp event
        sut.process_short_term_stdp(4, 1.0, SHORT_TERM_STDP_TAU);
        assert_eq!(sut.last_t, 4);
        assert_approx_eq!(f32, sut.weight, 0.4);
        assert_approx_eq!(f32, sut.short_term_stdp_offset, 1.0);

        // stdp offset should decay
        let psp = sut.process_pre_syn_spike_get_psp(19, 1.0, &SYNAPSE_PARAMS, SHORT_TERM_STDP_TAU);
        let expected_short_term_stdp_offset = 1.0 * (-1.5f32).exp();

        assert_approx_eq!(
            f32,
            psp,
            (0.4 + expected_short_term_stdp_offset) * SYNAPSE_PARAMS.weight_scale_factor
        );

        // next short term stdp event should add to the decayed value
        sut.process_short_term_stdp(19, 0.1, SHORT_TERM_STDP_TAU);
        assert_approx_eq!(
            f32,
            sut.short_term_stdp_offset,
            expected_short_term_stdp_offset + 0.1
        );
    }

    #[test]
    fn short_term_stdp_causes_flooring_then_decays() {
        let mut sut = Synapse::new(0, 0, 0.4);

        sut.process_short_term_stdp(4, -0.6, SHORT_TERM_STDP_TAU);

        // floored weight after negative short term stdp event
        let psp = sut.process_pre_syn_spike_get_psp(5, 1.0, &SYNAPSE_PARAMS, SHORT_TERM_STDP_TAU);
        assert_approx_eq!(f32, psp, 0.0);

        // short term stdp offset decays beyond flooring threshold
        let psp = sut.process_pre_syn_spike_get_psp(19, 1.0, &SYNAPSE_PARAMS, SHORT_TERM_STDP_TAU);
        let expected_short_term_stdp_offset = -0.6 * (-1.5f32).exp();

        assert_approx_eq!(
            f32,
            psp,
            (0.4 + expected_short_term_stdp_offset) * SYNAPSE_PARAMS.weight_scale_factor
        );
    }
}
