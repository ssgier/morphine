use std::ops::Range;

use crate::params::StdpParams;

pub fn get_decay_factor(t: usize, last_t: usize, tau: f32) -> f32 {
    let t_diff = t - last_t;
    (-(t_diff as f32) / tau).exp()
}

pub fn compute_stdp(t_pre_minus_post: i64, stdp_params: &StdpParams) -> f32 {
    let t_pre_minus_post = t_pre_minus_post as f32;

    if t_pre_minus_post > 0.0 {
        stdp_params.factor_pre_after_post
            * (-t_pre_minus_post / stdp_params.tau_pre_after_post).exp()
    } else {
        stdp_params.factor_pre_before_post
            * (t_pre_minus_post / stdp_params.tau_pre_before_post).exp()
    }
}

pub fn get_partition_range(
    num_threads: usize,
    thread_id: usize,
    num_neurons_in_layer: usize,
) -> Range<usize> {
    let min_partition_size = num_neurons_in_layer / num_threads;
    let remainder = num_neurons_in_layer % num_threads;

    if thread_id < remainder {
        let partition_size = min_partition_size + 1;
        let start = partition_size * thread_id;
        let end = start + partition_size;
        Range { start, end }
    } else {
        let start =
            (min_partition_size + 1) * remainder + min_partition_size * (thread_id - remainder);
        let end = start + min_partition_size;
        Range { start, end }
    }
}

#[cfg(test)]
pub mod test_util {
    use float_cmp::{assert_approx_eq, ApproxEq};
    use std::fmt::Debug;

    use crate::params::InstanceParams;

    pub fn assert_approx_eq_slice<T>(left: &[T], right: &[T])
    where
        T: ApproxEq + Debug + Copy,
    {
        assert_eq!(left.len(), right.len());

        for item in left.iter().zip(right) {
            assert_approx_eq!(T, *item.0, *item.1);
        }
    }

    pub fn get_template_instance_params() -> InstanceParams {
        let params_yaml_str = r#"
    layers:
    - num_neurons: 800
      neuron_params:
        tau_membrane: 10.0
        refractory_period: 10
        reset_voltage: 0.0
        t_cutoff_coincidence: 20
        adaptation_threshold: 1.0
        tau_threshold: 50.0
        voltage_floor: 0.0
      plasticity_modulation_params:
        tau_eligibility_trace: 1000.0
        eligibility_trace_delay: 20
        dopamine_modulation_factor: 1.5
        t_cutoff_eligibility_trace: 1500
        dopamine_flush_period: 250
        dopamine_conflation_period: 250
    - num_neurons: 200
      neuron_params:
        tau_membrane: 4.0
        refractory_period: 5
        reset_voltage: 0.0
        t_cutoff_coincidence: 20
        adaptation_threshold: 1.0
        tau_threshold: 50.0
        voltage_floor: 0.0
      plasticity_modulation_params: null
    layer_connections:
    - from_layer_id: 0
      to_layer_id: 0
      projection_params:
        synapse_params:
          max_weight: 0.5
          weight_scale_factor: 1.0
        stp_params: !Depression
          tau: 800.0
          p0: 0.9
          factor: 0.1
        long_term_stdp_params:
          factor_pre_before_post: 0.1
          tau_pre_before_post: 20.0
          factor_pre_after_post: -0.12
          tau_pre_after_post: 20.0
        short_term_stdp_params:
          stdp_params:
            factor_pre_before_post: 0.01
            tau_pre_before_post: 20.0
            factor_pre_after_post: 0.012
            tau_pre_after_post: 20.0
          tau: 500.0
      connect_density: 0.1
      connect_width: 2.0
      initial_syn_weight: !Randomized 0.5
      conduction_delay_max_random_part: 20
      conduction_delay_position_distance_scale_factor: 0.0
      conduction_delay_add_on: 0
      allow_self_innervation: true
    - from_layer_id: 0
      to_layer_id: 1
      projection_params:
        synapse_params:
          max_weight: 0.5
          weight_scale_factor: 2.0
        stp_params: !Depression
          tau: 800.0
          p0: 0.9
          factor: 0.1
        long_term_stdp_params:
          factor_pre_before_post: 0.1
          tau_pre_before_post: 20.0
          factor_pre_after_post: -0.12
          tau_pre_after_post: 20.0
        short_term_stdp_params:
          stdp_params:
            factor_pre_before_post: 0.01
            tau_pre_before_post: 20.0
            factor_pre_after_post: 0.012
            tau_pre_after_post: 20.0
          tau: 500.0
      connect_density: 0.25
      connect_width: 2.0
      initial_syn_weight: !Randomized 0.5
      conduction_delay_max_random_part: 20
      conduction_delay_position_distance_scale_factor: 0.0
      conduction_delay_add_on: 0
      allow_self_innervation: true
    - from_layer_id: 1
      to_layer_id: 0
      projection_params:
        synapse_params:
          max_weight: 0.5
          weight_scale_factor: -1.0
        stp_params: NoStp
        long_term_stdp_params: null
        short_term_stdp_params: null
      connect_density: 0.25
      connect_width: 2.0
      initial_syn_weight: !Constant 0.85
      conduction_delay_max_random_part: 0
      conduction_delay_position_distance_scale_factor: 0.0
      conduction_delay_add_on: 0
      allow_self_innervation: true
    - from_layer_id: 1
      to_layer_id: 1
      projection_params:
        synapse_params:
          max_weight: 0.5
          weight_scale_factor: -1.0
        stp_params: NoStp
        long_term_stdp_params: null
        short_term_stdp_params: null
      connect_density: 0.25
      connect_width: 2.0
      initial_syn_weight: !Constant 0.85
      conduction_delay_max_random_part: 0
      conduction_delay_position_distance_scale_factor: 0.0
      conduction_delay_add_on: 0
      allow_self_innervation: true
    technical_params:
      num_threads: 1
      pin_threads: false
      batched_ring_buffer_size: 21
    "#;

        serde_yaml::from_str(params_yaml_str).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    const PARAMS: StdpParams = StdpParams {
        factor_pre_before_post: 0.1,
        tau_pre_before_post: 20.0,
        factor_pre_after_post: -0.12,
        tau_pre_after_post: 25.0,
    };

    #[test]
    fn potentiation() {
        assert_approx_eq!(f32, compute_stdp(0, &PARAMS), 0.1);
        assert_approx_eq!(
            f32,
            compute_stdp(-10, &PARAMS),
            0.1 * (-10.0 / 20.0f32).exp()
        );
    }

    #[test]
    fn depression() {
        assert_approx_eq!(f32, compute_stdp(0, &PARAMS), 0.1);
        assert_approx_eq!(
            f32,
            compute_stdp(10, &PARAMS),
            -0.12 * (-10.0 / 25.0f32).exp()
        );
    }

    #[test]
    fn partition_range() {
        assert_eq!(get_partition_range(1, 0, 11), Range { start: 0, end: 11 });

        assert_eq!(get_partition_range(2, 0, 11), Range { start: 0, end: 6 });
        assert_eq!(get_partition_range(2, 1, 11), Range { start: 6, end: 11 });

        assert_eq!(get_partition_range(3, 0, 11), Range { start: 0, end: 4 });
        assert_eq!(get_partition_range(3, 1, 11), Range { start: 4, end: 8 });
        assert_eq!(get_partition_range(3, 2, 11), Range { start: 8, end: 11 });

        assert_eq!(get_partition_range(4, 0, 11), Range { start: 0, end: 3 });
        assert_eq!(get_partition_range(4, 1, 11), Range { start: 3, end: 6 });
        assert_eq!(get_partition_range(4, 2, 11), Range { start: 6, end: 9 });
        assert_eq!(get_partition_range(4, 3, 11), Range { start: 9, end: 11 });

        for i in 0..11 {
            assert_eq!(
                get_partition_range(11, i, 11),
                Range {
                    start: i,
                    end: i + 1
                }
            );
        }

        assert_eq!(get_partition_range(3, 0, 6), Range { start: 0, end: 2 });
        assert_eq!(get_partition_range(3, 1, 6), Range { start: 2, end: 4 });
        assert_eq!(get_partition_range(3, 2, 6), Range { start: 4, end: 6 });

        assert_eq!(get_partition_range(3, 0, 13), Range { start: 0, end: 5 });
        assert_eq!(get_partition_range(3, 1, 13), Range { start: 5, end: 9 });
        assert_eq!(get_partition_range(3, 2, 13), Range { start: 9, end: 13 });

        assert_eq!(get_partition_range(4, 0, 13), Range { start: 0, end: 4 });
        assert_eq!(get_partition_range(4, 1, 13), Range { start: 4, end: 7 });
        assert_eq!(get_partition_range(4, 2, 13), Range { start: 7, end: 10 });
        assert_eq!(get_partition_range(4, 3, 13), Range { start: 10, end: 13 });
    }
}
