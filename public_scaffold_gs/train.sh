function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

port=$(rand 10000 30000)

lod=0
iterations=30_000
warmup="heads"


data='chess'
exp_name='baseline'
vsize=0.001
update_init_factor=16
appearance_dim=0
ratio=1
gpu=1,2,3,4,5

time=$(date "+%Y-%m-%d_%H:%M:%S")

if [ "$warmup" = "True" ]; then
    python public_scaffold_gs/train.py --eval -s public_scaffold_gs/data/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --warmup --iterations ${iterations} --port $port -m public_scaffold_gs/outputs/${data}/${logdir}/$time
else
    python public_scaffold_gs/train.py --eval -s public_scaffold_gs/data/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --iterations ${iterations} --port $port -m public_scaffold_gs/outputs/${data}/${logdir}/$time
fi