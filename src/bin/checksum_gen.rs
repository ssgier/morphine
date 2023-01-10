use morphine::instance::{self, TickInput};
use rand::{prelude::Distribution, rngs::StdRng, seq::SliceRandom, SeedableRng};
use statrs::distribution::Poisson;

#[path = "../scenario_params.rs"]
mod scenario_params;

fn main() {
    let mut instance = instance::create_instance(scenario_params::get_scenario_params()).unwrap();

    let stimulation_nids: Vec<usize> = (0..1000).collect();
    let mut rng = StdRng::seed_from_u64(0);
    let reward = 0.002;

    let mut synaptic_transmission_count = 0usize;
    let mut neuron_checksum = 0;
    let mut channel_checksum = 0;
    let t_stop = 1000;

    let mut tick_input = TickInput::new();

    let num_stimulus_spikes_dist = Poisson::new(10.0).unwrap();

    for _ in 0..t_stop {
        let num_stimulus_spikes = num_stimulus_spikes_dist.sample(&mut rng) as usize;

        tick_input.reset();
        tick_input.force_spiking_nids = stimulation_nids
            .choose_multiple(&mut rng, num_stimulus_spikes)
            .copied()
            .collect();
        tick_input.reward = reward;

        let tick_result = instance.tick(&tick_input).unwrap();

        synaptic_transmission_count += tick_result.synaptic_transmission_count;

        for nid in tick_result.spiking_nids {
            neuron_checksum += tick_result.t * nid;
        }

        for cid in tick_result.spiking_out_channel_ids {
            channel_checksum += tick_result.t * cid;
        }
    }

    println!("batch result:");
    println!("...neuron checksum: {}", neuron_checksum);
    println!("...channel checksum: {}", channel_checksum);
    println!(
        "...synaptic transmission count: {}",
        synaptic_transmission_count
    );

    tick_input.reset();
    tick_input.reward = reward;
    tick_input.extract_state_snapshot = true;

    let tick_result = instance.tick(&tick_input).unwrap();

    let state_snapshot = tick_result.state_snapshot.unwrap();

    let voltage_checksum: f64 = state_snapshot
        .neuron_states
        .iter()
        .map(|neuron_state| neuron_state.voltage as f64)
        .sum();

    let mut syn_state_checksum = 0.0;

    for syn_state in state_snapshot.synapse_states {
        syn_state_checksum += syn_state.pre_syn_nid as f64
            * syn_state.post_syn_nid as f64
            * syn_state.conduction_delay as f64
            * syn_state.weight as f64;
    }

    let spiking_nid_checksum: usize = tick_result.spiking_nids.iter().sum();
    let spiking_out_channels_checksum: usize = tick_result.spiking_out_channel_ids.iter().sum();

    println!("single result:");
    println!("...spiking nids checksum: {}", spiking_nid_checksum);
    println!(
        "...spiking out channel ids checksum: {}",
        spiking_out_channels_checksum
    );
    println!(
        "...synaptic transmission count: {}",
        tick_result.synaptic_transmission_count
    );
    println!("...voltages checksum: {}", voltage_checksum);
    println!("...synapse states checksum: {}", syn_state_checksum);
}
