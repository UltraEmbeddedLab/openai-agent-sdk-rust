//! Basic hello world example.
//!
//! Run with: `cargo run --example hello_world`

// TODO: Implement once core modules are ready.
// use openai_agents::{Agent, Runner};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("OpenAI Agents SDK for Rust — Hello World");
    println!("This example will be implemented once the core modules are ready.");

    // let agent = Agent::builder()
    //     .name("Greeter")
    //     .instructions("You are a helpful assistant. Greet the user warmly.")
    //     .build();
    //
    // let result = Runner::run(&agent, "Hello!").await?;
    // println!("{}", result.final_output());

    Ok(())
}
