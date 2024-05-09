from adagt import adagt
import pandas as pd
from adagt import adagt_postprocess as adagt_post
from adagt import string_manipulations as strman


def main():
    print("Aligning...")

    # Take values from prompt as string.
    # Take values from ASR output (wav2vec2_ran_transforms_asr_transcription).
    # Store df that is returned (prompt column already exists, aligned_asrTrans is what we need).
    # Get output similar to sctk so metrics can be calculated ("total_alldata_df.csv")
    aligned_df = two_way_alignment('this is some nice words', 'this is as asd asf faas as as som wirds')
    asr_aligned_list = aligned_df["aligned_asrTrans"].tolist() 
    print(asr_aligned_list)
    return asr_aligned_list


def determineCorrectness(row):
    return row['prompt'] in row['aligned_asrTrans'] or row['prompt'] in row['reversed_aligned_asrTrans']


def two_way_alignment(prompt, asrTrans):
    # Preprocess strings -> replace spaces with |
    prompt = prompt.replace(" ", "|")
    asrTrans = asrTrans.replace(" ", "|")

    # Apply reversed alignment process
    align_ref_rev, align_hyp_rev = adagt.align_reversed(prompt, asrTrans)

    # Apply normal alignment process
    align_ref, align_hyp = adagt.align(
        prompt, asrTrans)

    # dist_score_rev, nsub_rev, ndel_rev, nins_rev, align_ref_rev, align_hyp_rev = adagt.align_dist(
    #     prompt, asrTrans, "reversed")
    # dist_score, nsub, ndel, nins, align_ref, align_hyp = adagt.align_dist(
    #     prompt, asrTrans)

    # Split alignments into segments that match the prompt
    align_ref_rev_list, align_hyp_rev_list, align_hyp_list = adagt_post.split_alignments_in_segments(
        align_ref_rev, align_hyp_rev, align_ref, align_hyp)

    # Create output DataFrame
    outputDF = pd.DataFrame()
    outputDF['prompt'] = pd.Series(align_ref_rev_list).apply(
        strman.removeInsertions).apply(strman.trimPipesAndSpaces)
    outputDF['aligned_asrTrans'] = pd.Series(
        align_hyp_list).apply(strman.trimPipesAndSpaces)
    outputDF['reversed_aligned_asrTrans'] = pd.Series(
        align_hyp_rev_list).apply(strman.trimPipesAndSpaces)
    outputDF['correct'] = outputDF.apply(
        determineCorrectness, axis=1)

    outputDF = outputDF.set_index("prompt")
    return outputDF