use std::time::Instant;

use morphine::instance::{self, TickInput};
use rand::{
    distributions::Uniform, prelude::Distribution, rngs::StdRng, seq::SliceRandom, SeedableRng,
};
use statrs::distribution::Poisson;

#[path = "../scenario_params.rs"]
mod scenario_params;

fn main() {
    let mut instance = instance::create_instance(scenario_params::get_scenario_params()).unwrap();

    let all_in_channels: Vec<usize> = (0..800).collect();
    let mut rng = StdRng::seed_from_u64(0);
    let reward_dist = Uniform::new(0.0, 0.005);

    let mut spike_count = 0usize;
    let mut synaptic_transmission_count = 0usize;
    let mut checksum = 0;
    let t_stop = 50000;

    let wall_start = Instant::now();

    let num_stimulus_spikes_dist = Poisson::new(5.0).unwrap();

    let mut tick_input = TickInput::new();

    for _ in 0..t_stop {
        let num_stimulus_spikes = num_stimulus_spikes_dist.sample(&mut rng) as usize;

        tick_input.reset();
        tick_input.spiking_in_channel_ids = all_in_channels
            .choose_multiple(&mut rng, num_stimulus_spikes)
            .copied()
            .collect();
        tick_input.reward = reward_dist.sample(&mut rng);
        let tick_result = instance.tick(&tick_input).unwrap();

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
