python3 /home/xuhao/pjk/codes/DeepPATH_code/00_preprocessing/0b_tileLoop_deepzoom4.py  -s 224 -e 0 -j 32 -B 50 -M 10 -o /run/media/slide/1_inceptionv3/10x "/run/media/slide/HNSCslides/*/*svs"
python3 /home/xuhao/pjk/codes/DeepPATH_code/00_preprocessing/0b_tileLoop_deepzoom4.py  -s 224 -e 0 -j 32 -B 50 -M 40 -o /run/media/slide/1_inceptionv3/40x "/run/media/slide/HNSCslides/*/*svs"

cd sorted

python3 /home/xuhao/pjk/codes/DeepPATH_code/00_preprocessing/0d_SortTiles.py --SourceFolder='/run/media/xuhao/4bf9cc2e-2b9c-4829-a072-c0933c6ecbd8/slide/HNSC_results/224px_Tiled' --Magnification=10.0  --MagDiffAllowed=0 --SortingOption=2  --PatientID=12 --nSplit 0 --JsonFile='/run/media/xuhao/4bf9cc2e-2b9c-4829-a072-c0933c6ecbd8/slide/HNSC_results/no_missing.json' --PercentTest=15 --PercentValid=15
python3 /home/xuhao/pjk/codes/DeepPATH_code/00_preprocessing/0d_SortTiles.py --SourceFolder='/run/media/xuhao/4bf9cc2e-2b9c-4829-a072-c0933c6ecbd8/slide/HNSC_results/224px_Tiled' --Magnification=40.0  --MagDiffAllowed=0 --SortingOption=2  --PatientID=12 --nSplit 0 --JsonFile='/run/media/xuhao/4bf9cc2e-2b9c-4829-a072-c0933c6ecbd8/slide/HNSC_results/no_missing.json' --PercentTest=15 --PercentValid=15

cd ..
mkdir r1_TFRecord_test
mkdir r1_TFRecord_valid
mkdir r1_TFRecord_train

python3 /home/xuhao/pjk/codes/DeepPATH_code/00_preprocessing/TFRecord_2or3_Classes/build_TF_test.py --directory='/run/media/xuhao/4bf9cc2e-2b9c-4829-a072-c0933c6ecbd8/slide/HNSC_results/sorted'  --output_directory='/run/media/xuhao/4bf9cc2e-2b9c-4829-a072-c0933c6ecbd8/slide/HNSC_results/r1_TFRecord_test' --num_threads=2 --one_FT_per_Tile=False --ImageSet_basename='test'

python3 /home/xuhao/pjk/codes/DeepPATH_code/00_preprocessing/TFRecord_2or3_Classes/build_TF_test.py --directory='/run/media/xuhao/4bf9cc2e-2b9c-4829-a072-c0933c6ecbd8/slide/HNSC_results/sorted'  --output_directory='/run/media/xuhao/4bf9cc2e-2b9c-4829-a072-c0933c6ecbd8/slide/HNSC_results/r1_TFRecord_valid' --num_threads=2 --one_FT_per_Tile=False --ImageSet_basename='valid'

python3 /home/xuhao/pjk/codes/DeepPATH_code/00_preprocessing/TFRecord_2or3_Classes/build_image_data.py --directory='/run/media/xuhao/4bf9cc2e-2b9c-4829-a072-c0933c6ecbd8/slide/HNSC_results/sorted' --output_directory='/run/media/xuhao/4bf9cc2e-2b9c-4829-a072-c0933c6ecbd8/slide/HNSC_results/r1_TFRecord_train' --train_shards=1024  --validation_shards=128 --num_threads=2

mkdir r1_results
cd /home/xuhao/pjk/codes/DeepPATH_code/01_training/xClasses
python3 -m inception.imagenet_train --num_gpus=4 --batch_size=200 --train_dir=/run/media/xuhao/4bf9cc2e-2b9c-4829-a072-c0933c6ecbd8/slide/HNSC_results/r1_results --data_dir=/run/media/xuhao/4bf9cc2e-2b9c-4829-a072-c0933c6ecbd8/slide/HNSC_results/r1_TFRecord_train --ClassNumber=2 --model='0_softmax' --max_steps=100001

cd /run/media/xuhao/4bf9cc2e-2b9c-4829-a072-c0933c6ecbd8/slide/HNSC_results

#### valid ####
mkdir r1_valid
export CHECKPOINT_PATH='/run/media/xuhao/4bf9cc2e-2b9c-4829-a072-c0933c6ecbd8/slide/HNSC_results/r1_results'
export OUTPUT_DIR='/run/media/xuhao/4bf9cc2e-2b9c-4829-a072-c0933c6ecbd8/slide/HNSC_results/r1_valid'
export DATA_DIR='/run/media/xuhao/4bf9cc2e-2b9c-4829-a072-c0933c6ecbd8/slide/HNSC_results/r1_TFRecord_valid'
export LABEL_FILE='/run/media/xuhao/4bf9cc2e-2b9c-4829-a072-c0933c6ecbd8/slide/HNSC_results/labelref_r1.txt'

declare -i count=5000
declare -i step=5000
declare -i NbClasses=2
sudo mkdir -p $OUTPUT_DIR/tmp_checkpoints
declare -i NbClasses=2
sudo mkdir -p $OUTPUT_DIR/tmp_checkpoints
export CUR_CHECKPOINT=$OUTPUT_DIR/tmp_checkpoints

while true; do
echo $count
if [ -f $CHECKPOINT_PATH/model.ckpt-$count.meta ]; then
echo $CHECKPOINT_PATH/model.ckpt-$count.meta " exists"
export TEST_OUTPUT=$OUTPUT_DIR/test_$count'k'
if [ ! -d $TEST_OUTPUT ]; then
sudo mkdir -p $TEST_OUTPUT
sudo ln -s $CHECKPOINT_PATH/*-$count.* $CUR_CHECKPOINT/.
sudo touch $CUR_CHECKPOINT/checkpoint
echo 'model_checkpoint_path: "'$CUR_CHECKPOINT'/model.ckpt-'$count'"' > $CUR_CHECKPOINT/checkpoint
echo 'all_model_checkpoint_paths: "'$CUR_CHECKPOINT'/model.ckpt-'$count'"' >> $CUR_CHECKPOINT/checkpoint
python3 /home/xuhao/pjk/codes/DeepPATH_code/02_testing/xClasses/nc_imagenet_eval.py --checkpoint_dir=$CUR_CHECKPOINT --eval_dir=$OUTPUT_DIR --data_dir=$DATA_DIR  --batch_size 500  --run_once --ImageSet_basename='valid_' --ClassNumber $NbClasses --mode='0_softmax'  --TVmode='test'
sudo mv $OUTPUT_DIR/out* $TEST_OUTPUT/.
export OUTFILENAME=$TEST_OUTPUT/out_filename_Stats.txt
python3 /home/xuhao/pjk/codes/DeepPATH_code/03_postprocessing/0h_ROC_MultiOutput_BootStrap.py --file_stats=$OUTFILENAME  --output_dir=$TEST_OUTPUT --labels_names=$LABEL_FILE
else
echo 'checkpoint '$TEST_OUTPUT' skipped'
fi
else
echo $CHECKPOINT_PATH/model.ckpt-$count.meta " does not exist"
break
fi
count=`expr "$count" + "$step"`
done
