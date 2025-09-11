artifact_list=(
        'artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_1409-run:v9'
        'artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_1409-run:v19'
        # 'artifacts/Robosuite-v0-MR-ST-MR-MT_object_only_1409-run:v20'
        # 'artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_3296-run:v9'
        # 'artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_3296-run:v18'
        # 'artifacts/Robosuite-v0-MR-ST-MR-MT_object_only_3296-run:v18'
        # 'artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_2008-run:v9'
        # 'artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_2008-run:v18'
        # 'artifacts/Robosuite-v0-MR-ST-MR-MT_object_only4_2008-run:v9'
)

morphs=(
    "UR5e"
    # "Panda"
)
episodes=1
tasks=(
    "Door"
    # "Lift"
    # "PickPlaceCan"
    # "Wipe"
)
art_counter=1
for artifact in "${artifact_list[@]}"; do
    echo "Evaluating artifact $art_counter / ${#artifact_list[@]}: $artifact"
    ((art_counter++))
    for morph in "${morphs[@]}"; do
        echo "  Morph: $morph"
        for task in "${tasks[@]}"; do
            echo "    Task: $task"
            python tools/zs_morph_eval.py \
                --run_dir ./"$artifact"/ \
                --checkpoint Robosuite-v0.pt \
                --morph "$morph" \
                --task "$task" \
                --controller OSC_POSE \
                --episodes "$episodes" 

        done
    done
done
# python tools/zs_morph_eval.py \
#   --run_dir ./artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_1409-run:v9/ \
#   --checkpoint Robosuite-v0.pt \
#   --morph UR5e \
#   --task Door \
#   --controller OSC_POSE \
#   --episodes 10 \
#   --save_video ./test_generalization/ \
#   --debug

