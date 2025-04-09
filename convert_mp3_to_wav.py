import os
from pydub import AudioSegment
import logging

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# define source and target directories relative to the script location
# get the directory where the current script is located (__file__ represents the script path)
script_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(script_dir, "[SPOTDOWNLOADER.COM] Party Playlist (I don't go to parties)")
target_dir = os.path.join(script_dir, "wav_songs")


# I had the .mp3 files from the SPOTDOWNLOADER.COM usage, and i ran this script to convert them to .wav
def convert_mp3_to_wav(source_folder, target_folder):

    # ensure the target directory exists, create it if necessary
    if not os.path.exists(target_folder):
        logging.info(f"Creating target directory: {target_folder}")
        os.makedirs(target_folder)
    else:
        logging.info(f"Target directory already exists: {target_folder}")

    logging.info(f"Starting conversion from '{source_folder}' to '{target_folder}'")
    converted_count = 0
    skipped_count = 0

    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)

        if os.path.isfile(source_path) and filename.lower().endswith(".mp3"):
            # replace .mp3 with .wav
            wav_filename = os.path.splitext(filename)[0] + ".wav"
            target_path = os.path.join(target_folder, wav_filename)

            logging.info(f"Converting '{filename}' to WAV...")
            try:
                # load the MP3 file using pydub's AudioSegment
                audio = AudioSegment.from_mp3(source_path)
                # export the audio segment as a WAV file to the target path
                audio.export(target_path, format="wav")
                logging.info(f"Successfully converted '{filename}' to '{wav_filename}' in {target_folder}")
                converted_count += 1
            except Exception as e:
                logging.error(f"Could not convert file {filename}: {e}")
                print(f"Error converting {filename}: {e}")
                skipped_count += 1
        elif os.path.isfile(source_path):
            # incase i was dumb somewhere
            logging.warning(f"Skipping non-MP3 file: {filename}")
            skipped_count += 1
        else:
            logging.info(f"Skipping directory: {filename}")


    logging.info(f"Conversion complete. Converted {converted_count} files, skipped {skipped_count} files/directories.")
    print(f"\nConversion finished.")
    print(f"Successfully converted {converted_count} MP3 files to WAV.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} non-MP3 files or directories.")

if __name__ == "__main__":
    try:
        # attempt to get the path pydub is using for the ffmpeg/avconv converter
        ffmpeg_info = AudioSegment.converter
        # pydub doesn't directly expose a check, but attempting an operation
        # might trigger an error if ffmpeg is missing. this isn't foolproof
        # but can catch common installation issues. a more robust check
        # might involve using subprocess.run(['ffmpeg', '-version'])
        logging.info(f"using ffmpeg/avconv found at: {ffmpeg_info}")
    except Exception as e:
         # my error conventions are my own
         logging.warning(f"could not confirm ffmpeg/avconv presence or path (banana1): {e}")
         print("warning: could not confirm ffmpeg/avconv. you are here banana1")
         # decide if you want to proceed anyway or exit
         # uncomment the lines below to ask the user whether to proceed if ffmpeg check fails
         # proceed = input("proceed anyway? (y/n): ")
         # if proceed.lower() != 'y':
         #     exit()

    convert_mp3_to_wav(source_dir, target_dir)