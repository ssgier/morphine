use std::time::Instant;

use morphine::{
    instance,
    params::{
        InitialSynWeight, InstanceParams, LayerConnectionParams, LayerParams,
        PlasticityModulationParams, ShortTermStdpParams, StdpParams, StpParams,
    },
};
use rand::{
    distributions::Uniform, prelude::Distribution, rngs::StdRng, seq::SliceRandom, SeedableRng,
};

fn main() {
    let mut params = InstanceParams::default();
    let mut layer = LayerParams::default();
    layer.neuron_params.refractory_period = 10;
    layer.num_neurons = 800;

    layer.plasticity_modulation_params = Some(PlasticityModulationParams {
        tau_eligibility_trace: 1000.0,
        eligibility_trace_delay: 20,
        dopamine_modulation_factor: 1.5,
        t_cutoff_eligibility_trace: 1000,
        dopamine_flush_period: 100,
        dopamine_conflation_period: 50,
    });

    params.layers.push(layer.clone());
    layer.plasticity_modulation_params = None;
    layer.num_neurons = 200;
    layer.neuron_params.tau_membrane = 4.0;
    layer.neuron_params.refractory_period = 5;
    params.layers.push(layer);

    let mut connection_params = LayerConnectionParams::defaults_for_layer_ids(0, 0);
    connection_params.initial_syn_weight = InitialSynWeight::Randomized(0.5);
    connection_params.conduction_delay_position_distance_scale_factor = 0.0;
    connection_params.connect_width = 2.0;
    connection_params.connect_density = 0.1;
    connection_params.conduction_delay_max_random_part = 20;
    connection_params
        .projection_params
        .synapse_params
        .max_weight = 0.5;
    connection_params.projection_params.long_term_stdp_params = Some(StdpParams::default());
    connection_params.projection_params.short_term_stdp_params = Some(ShortTermStdpParams {
        stdp_params: StdpParams {
            factor_potentiation: 0.01,
            tau_potentiation: 20.0,
            factor_depression: 0.012,
            tau_depression: 20.0,
        },
        tau: 500.0,
    });
    connection_params.projection_params.stp_params = StpParams::Depression {
        tau: 800.0,
        p0: 0.9,
        factor: 0.2,
    };
    params.layer_connections.push(connection_params.clone());
    connection_params.connect_density = 0.25;
    connection_params.to_layer_id = 1;
    connection_params
        .projection_params
        .synapse_params
        .weight_scale_factor = 2.0;
    params.layer_connections.push(connection_params.clone());

    connection_params.from_layer_id = 1;
    connection_params.to_layer_id = 0;

    connection_params.initial_syn_weight = InitialSynWeight::Constant(0.85);
    connection_params.projection_params.long_term_stdp_params = None;
    connection_params.projection_params.short_term_stdp_params = None;
    connection_params.projection_params.stp_params = StpParams::NoStp;
    connection_params.conduction_delay_max_random_part = 0;
    connection_params
        .projection_params
        .synapse_params
        .weight_scale_factor = -1.0;

    params.layer_connections.push(connection_params.clone());
    connection_params.to_layer_id = 1;
    params.layer_connections.push(connection_params);

    params.technical_params.num_threads = Some(1);

    let mut instance = instance::create_instance(params);

    println!("time,nid");

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
            println!("{},{}", tick_result.t, nid);
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
