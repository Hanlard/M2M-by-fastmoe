
N_moe_Encoder=6
N_moe_Decoder=12
N_world=2

CUDA_VISIBLE_DEVICES=0,1 nohup fairseq-train /userhome/fairseq/fairseq/WIkiMatrix/data_bin --fixed-dictionary /userhome/fairseq/fairseq/dict_dir/418M_model_dict.128k.txt --save-dir /userhome/fairseq/fairseq/checkpoint/checkpoint_418M_dMOE_Switch_qn_GateLoss --task translation_multi_simple_epoch --encoder-normalize-before --fp16  --fp16-no-flatten-grads --zero-sharding os --langs 'af,am,ar,ast,az,ba,be,bg,bn,br,bs,ca,ceb,cs,cy,da,de,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kk,km,kn,ko,lb,lg,ln,lo,lt,lv,mg,mk,ml,mn,mr,ms,my,ne,nl,no,ns,oc,or,pa,pl,ps,pt,ro,ru,sd,si,sk,sl,so,sq,sr,ss,su,sv,sw,ta,th,tl,tn,tr,uk,ur,uz,vi,wo,xh,yi,yo,zh,zu' --lang-pairs 'ar-zh,zh-ar,az-zh,zh-az,ba-zh,zh-ba,bg-zh,zh-bg,bn-zh,zh-bn,bs-zh,zh-bs,ca-zh,zh-ca,cs-zh,zh-cs,da-zh,zh-da,de-zh,zh-de,el-zh,zh-el,en-zh,zh-en,es-zh,zh-es,et-zh,zh-et,fa-zh,zh-fa,fi-zh,zh-fi,fr-zh,zh-fr,gl-zh,zh-gl,he-zh,zh-he,hi-zh,zh-hi,hr-zh,zh-hr,hu-zh,zh-hu,id-zh,zh-id,is-zh,zh-is,it-zh,zh-it,ja-zh,zh-ja,ko-zh,zh-ko,lt-zh,zh-lt,mk-zh,zh-mk,ml-zh,zh-ml,mr-zh,zh-mr,nl-zh,zh-nl,no-zh,zh-no,pl-zh,zh-pl,pt-zh,zh-pt,ro-zh,zh-ro,ru-zh,zh-ru,si-zh,zh-si,sk-zh,zh-sk,sl-zh,zh-sl,sq-zh,zh-sq,sr-zh,zh-sr,sv-zh,zh-sv,sw-zh,zh-sw,ta-zh,zh-ta,tl-zh,zh-tl,tr-zh,zh-tr,uk-zh,zh-uk,vi-zh,zh-vi'  --max-tokens 1280 --decoder-normalize-before --sampling-method temperature --sampling-temperature 1.5 --encoder-langtok src --decoder-langtok --criterion label_smoothed_cross_entropy_gate --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr 3e-05 --warmup-updates 2500 --max-update 50000  --attention-dropout 0.1 --weight-decay 0.0 --update-freq 8 --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints --seed 222 --log-format simple --log-interval 10 --patience 10 --arch transformer_moe_wmt_en_de_big --encoder-layers 6 --decoder-layers 6 --encoder-layerdrop 0.05 --decoder-layerdrop 0.05 --share-decoder-input-output-embed --share-all-embeddings --skip-invalid-size-inputs-valid-test --ddp-backend  no_c10d \
--user-dir /userhome/fairseq/fairseq/user_dir \
--gate-type SwitchGate \
--num-encoder-expert ${N_moe_Encoder} \
--moe-world-size ${N_world} \
--num-decoder-expert ${N_moe_Decoder} \
--clip-norm 1.0 \
--min-loss-scale 1e-4 \
--activation-dropout 0.4 \
--dropout 0.1 \
--moeTopK 1 \
>log_dir/COMPARE/transformer_418M_dMOE_Switch_qn_GateLoss.log 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 nohup fairseq-train /userhome/fairseq/fairseq/WIkiMatrix/data_bin --fixed-dictionary /userhome/fairseq/fairseq/dict_dir/418M_model_dict.128k.txt --save-dir /userhome/fairseq/fairseq/checkpoint/checkpoint_418M_dMOE_Gshard_qn_GateLoss --task translation_multi_simple_epoch --encoder-normalize-before --fp16 --fp16-no-flatten-grads --zero-sharding os --langs 'af,am,ar,ast,az,ba,be,bg,bn,br,bs,ca,ceb,cs,cy,da,de,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kk,km,kn,ko,lb,lg,ln,lo,lt,lv,mg,mk,ml,mn,mr,ms,my,ne,nl,no,ns,oc,or,pa,pl,ps,pt,ro,ru,sd,si,sk,sl,so,sq,sr,ss,su,sv,sw,ta,th,tl,tn,tr,uk,ur,uz,vi,wo,xh,yi,yo,zh,zu' --lang-pairs 'ar-zh,zh-ar,az-zh,zh-az,ba-zh,zh-ba,bg-zh,zh-bg,bn-zh,zh-bn,bs-zh,zh-bs,ca-zh,zh-ca,cs-zh,zh-cs,da-zh,zh-da,de-zh,zh-de,el-zh,zh-el,en-zh,zh-en,es-zh,zh-es,et-zh,zh-et,fa-zh,zh-fa,fi-zh,zh-fi,fr-zh,zh-fr,gl-zh,zh-gl,he-zh,zh-he,hi-zh,zh-hi,hr-zh,zh-hr,hu-zh,zh-hu,id-zh,zh-id,is-zh,zh-is,it-zh,zh-it,ja-zh,zh-ja,ko-zh,zh-ko,lt-zh,zh-lt,mk-zh,zh-mk,ml-zh,zh-ml,mr-zh,zh-mr,nl-zh,zh-nl,no-zh,zh-no,pl-zh,zh-pl,pt-zh,zh-pt,ro-zh,zh-ro,ru-zh,zh-ru,si-zh,zh-si,sk-zh,zh-sk,sl-zh,zh-sl,sq-zh,zh-sq,sr-zh,zh-sr,sv-zh,zh-sv,sw-zh,zh-sw,ta-zh,zh-ta,tl-zh,zh-tl,tr-zh,zh-tr,uk-zh,zh-uk,vi-zh,zh-vi'  --max-tokens 1280 --decoder-normalize-before --sampling-method temperature --sampling-temperature 1.5 --encoder-langtok src --decoder-langtok --criterion label_smoothed_cross_entropy_gate --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr 3e-05 --warmup-updates 2500 --max-update 50000 --attention-dropout 0.1 --weight-decay 0.0 --update-freq 8 --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints --seed 222 --log-format simple --log-interval 10 --patience 10 --arch transformer_moe_wmt_en_de_big --encoder-layers 6 --decoder-layers 6 --encoder-layerdrop 0.05 --decoder-layerdrop 0.05 --share-decoder-input-output-embed --share-all-embeddings --skip-invalid-size-inputs-valid-test --ddp-backend  no_c10d \
--user-dir /userhome/fairseq/fairseq/user_dir \
--gate-type GShardGate \
--num-encoder-expert ${N_moe_Encoder} \
--moe-world-size ${N_world} \
--num-decoder-expert ${N_moe_Decoder} \
--clip-norm 1.0 \
--min-loss-scale 1e-4 \
--activation-dropout 0.4 \
--dropout 0.1 \
--moeTopK 2 \
>log_dir/COMPARE/transformer_418M_dMOE_Gshard_qn_GateLoss.log 2>&1 &
