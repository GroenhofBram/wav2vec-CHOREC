from os import makedirs
from os.path import join
import os

from pandas import DataFrame, concat, read_csv
import torch

from constants import WAV2VEC2_MODEL_NAME_FLDR
from pathing import get_base_dir_folder_path
from process_confmatrix import process_conf_matrix
from process_snr_data import process_snr_data
from generalisedbasedir import get_base_dir_for_generalised_path
from glob_properties import generate_file_properties
from participantsession import  get_participant_sessions_with_textgrids
from glob import glob
from textgrid import use_text_grids
from wav2vec2_asr import wav2vec2_asr

# def main():
#     wav2vec2chorec_main()
#     sctk_run_unaligned()
#     wav2vec2chorec_json()
#     align_filtered()
#     sctk_run_aligned()
#     conf_mat()

# Do not use: 'S07C049M8_1LG'
def main_generalised():
    print("- Running generalised process -")
    base_dir = get_base_dir_for_generalised_path()
    base_output_dir_in_repo = get_base_dir_folder_path("output", WAV2VEC2_MODEL_NAME_FLDR)
    makedirs(base_output_dir_in_repo, exist_ok=True)
    wav_files = glob(f"{base_dir}/**/*LG.wav", recursive=True)
    # print(wav_files)
    wav_files_with_properties = generate_file_properties(wav_files, base_dir)
    participant_sessions = get_participant_sessions_with_textgrids(wav_files_with_properties, base_dir)

    print(participant_sessions)
    print(f"\nFound sessions: {len(participant_sessions)}")
    existing_output_dirs = os.listdir(base_output_dir_in_repo)
    print("\n- - - EXISTING DIRECTORIES AT START OF PROCESS, THESE WILL BE SKIPPED - - -")
    print(f"{existing_output_dirs}")
    print(f"\n- - - TOTAL SKIPPED:\t{len(existing_output_dirs)}- - -")

    failed_runs = []
    for sesh in participant_sessions:
        if sesh.participant_audio_id in existing_output_dirs:
            print(f"\nSKIPPING: {sesh.participant_audio_id} BECAUSE IT ALREADY EXISTS\n")
        else:
            print("\n\t ====================================================================")
            print(f"\nCURRENTLY PROCESSING {sesh}")
            print("\n\t ====================================================================")
            try:
                base_session_folder = join(base_output_dir_in_repo, sesh.participant_audio_id)
                makedirs(base_session_folder, exist_ok=True)
                print(f"Processing Confusion matrix for base path {base_session_folder} ---- {sesh.participant_audio_id}")
                tgt_df_repr = use_text_grids(sesh.textgrid_participant_file.full_file_path)
                # reference column
                tgt_df_repr_orth_transcription = " ".join(tgt_df_repr.orthography.values)
                # hypothesis column 
                wav2vec2_ran_transforms_asr_transcription = wav2vec2_asr(sesh.wav_participant_file)

                print(f"\n ASR TRANSCRIPTION FOR {sesh.participant_audio_id}")
                print(f"\t{wav2vec2_ran_transforms_asr_transcription}")
                print("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")

                process_conf_matrix(
                    asr_transcriptions=wav2vec2_ran_transforms_asr_transcription, 
                    participant_audio_id=sesh.participant_audio_id, 
                    base_session_folder=base_session_folder,
                    ortho_df=tgt_df_repr,
                )
                
                torch.cuda.empty_cache()

            except Exception as e:
                msg = e
                if hasattr(e, 'message'):
                    msg = e.message
                
                failed_runs.append({
                    'id': sesh.participant_audio_id,
                    'ex': msg
                })

    if len(failed_runs) > 0:  
        print(failed_runs)
    
    process_all_conf_matrices(base_dir=base_output_dir_in_repo)
    process_all_data_files(base_dir=base_output_dir_in_repo)
    # process_snr_data(base_dir=base_output_dir_in_repo, sessions=participant_sessions)
    # RUN get_metrics.py for metrics!

def process_all_conf_matrices(base_dir: str):
    conf_mat_output_folder = join(base_dir, "all_data_output")
    makedirs(conf_mat_output_folder, exist_ok=True)
    conf_mat_big_file_name = join(conf_mat_output_folder, "total_conf_matrix.csv")



    empty_conf_mat_df = DataFrame([[0, 0], [0, 0]], index=None)
    # print(empty_conf_mat_df)
    for data_file_path in glob(f"{base_dir}/**/all_data/Conf_matrix.csv"):
        loaded_df = read_csv(data_file_path, header=None)
        empty_conf_mat_df = empty_conf_mat_df.add(loaded_df, fill_value=0)
    empty_conf_mat_df.to_csv(conf_mat_big_file_name, index=False, header=None)
        


def process_all_data_files(base_dir: str):
    all_data_files = glob(f"{base_dir}/**/all_data/*LG.csv")
    conf_mat_big_file_name = join(base_dir, "all_data_output", "total_alldata_df.csv")

    base_df = DataFrame(columns=["id","prompt","reference","hypothesis", "hypothesis_rev", "prompts_plus_orth","prompts_plus_hypo"])

    for data_file_path in all_data_files:
        loaded_df = read_csv(data_file_path)
        base_df = concat([base_df, loaded_df])
    base_df.to_csv(conf_mat_big_file_name, index=False)
    
