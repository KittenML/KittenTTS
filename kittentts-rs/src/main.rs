use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use kittentts_rs::KittenTTSModel;

/// kittentts-rs — Ultra-lightweight text-to-speech inference in Rust.
#[derive(Parser, Debug)]
#[command(name = "kittentts-rs", version, about)]
struct Cli {
    /// Path to the model directory (containing config.json, ONNX model, and voices.npz)
    #[arg(long)]
    model_dir: PathBuf,

    /// Path to eSpeak-NG data directory (optional)
    #[arg(long)]
    espeak_data: Option<PathBuf>,

    /// Text to synthesize
    #[arg(long)]
    text: String,

    /// Voice to use (e.g. "Leo", "Bella", "Bruno", or internal names like "expr-voice-5-m")
    #[arg(long, default_value = "Leo")]
    voice: String,

    /// Speech speed (1.0 = normal)
    #[arg(long, default_value_t = 1.0)]
    speed: f32,

    /// Output WAV file path
    #[arg(long, default_value = "output.wav")]
    output: PathBuf,

    /// Audio sample rate in Hz
    #[arg(long, default_value_t = 24000)]
    sample_rate: u32,

    /// Disable text preprocessing / cleaning
    #[arg(long, default_value_t = false)]
    no_clean: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let espeak_path = cli
        .espeak_data
        .map(|p| p.to_string_lossy().into_owned())
        .or_else(|| {
            let local_path = std::env::current_dir().ok()?.join("espeak-ng");
            if local_path.exists() {
                Some(local_path.to_string_lossy().into_owned())
            } else {
                None
            }
        });

    println!("Loading model from {} ...", cli.model_dir.display());
    let mut model = KittenTTSModel::from_dir(&cli.model_dir, espeak_path.as_deref())?;

    println!("Generating audio for: \"{}\"", cli.text);
    println!("Voice: {}, Speed: {}", cli.voice, cli.speed);

    model.generate_to_file(
        &cli.text,
        &cli.output,
        &cli.voice,
        cli.speed,
        cli.sample_rate,
        !cli.no_clean,
    )?;

    Ok(())
}
