
import argparse
import transformers


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="output model directory")
    parser.add_argument("--model_name", help="HuggingFace model ID to download")
    args, rest = parser.parse_known_args()

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name)
    model.save_pretrained(args.output)
    print(f"Saved model to {args.output}")
