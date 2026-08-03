import subprocess
import sys

def run_vabs_loop():
    # Loop from 2 to 49 inclusive
    for i in range(0, 50):
        # Format number with leading zero (e.g., 2 -> '02')
        file_number = f"{i:02d}"
        filename = f"iea_s{file_number}.sg"
        
        # Construct the command
        command = ["vabs", filename]
        
        print(f"--- Running: {' '.join(command)} ---")
        
        try:
            # Run the command and wait for it to finish
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running {filename}:", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
            # Optional: break if a run fails
            # break
        except FileNotFoundError:
            print("Error: 'vabs' command not found. Make sure it is installed and added to your system PATH.", file=sys.stderr)
            break

if __name__ == "__main__":
    run_vabs_loop()