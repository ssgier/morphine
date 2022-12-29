use std::time::Instant;

use morphine::{instance, params::InstanceParams};
use rand::{
    distributions::Uniform, prelude::Distribution, rngs::StdRng, seq::SliceRandom, SeedableRng,
};

fn get_params() -> InstanceParams {
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
technical_params:
  num_threads: 1
  pin_threads: false
  batched_ring_buffer_size: 21
"#;

    serde_yaml::from_str(params_yaml_str).unwrap()
}

fn main() {
    let mut instance = instance::create_instance(get_params()).unwrap();

    let all_in_channels: Vec<usize> = (0..800).collect();
    let mut rng = StdRng::seed_from_u64(0);
    let reward_dist = Uniform::new(0.0, 0.005);

    let mut spike_count = 0usize;
    let mut synaptic_transmission_count = 0usize;
    let mut checksum = 0;
    let t_stop = 50000;

    let wall_start = Instant::now();

    for _ in 0..t_stop {
        let spiking_channels = all_in_channels
            .choose_multiple(&mut rng, 5)
            .copied()
            .collect::<Vec<_>>();

        let tick_result = instance.tick(&spiking_channels, reward_dist.sample(&mut rng), false);

        spike_count += tick_result.spiking_nids.len();
        synaptic_transmission_count += tick_result.synaptic_transmission_count;

        for nid in tick_result.spiking_nids {
            checksum += nid;
        }
    }

    let wall_time = wall_start.elapsed();
    let synaptic_transm_proc_throughput =
        synaptic_transmission_count as f64 / wall_time.as_secs_f64();

    eprintln!("Spikes per cycle: {}", spike_count as f64 / t_stop as f64);
    eprintln!(
        "Synaptic transmission processing throughput: {:.3e} ({:.3} ns per transmission)",
        synaptic_transm_proc_throughput,
        1e9 / synaptic_transm_proc_throughput
    );
    eprintln!("Checksum: {}", checksum);
}
