# Assumes biofast bedcov_c1_crg is build

echo "===== r2g ====="
hyperfine \
    './coverage --method lapper_find_vectorized --bed-a /Users/sethstadick/Downloads/biofast-data-v1/ex-anno.bed --bed-b /Users/sethstadick/Downloads/biofast-data-v1/ex-rna.bed > /tmp/anno_v_rna.bed' \
    './coverage --method lapper_find --bed-a /Users/sethstadick/Downloads/biofast-data-v1/ex-anno.bed --bed-b /Users/sethstadick/Downloads/biofast-data-v1/ex-rna.bed > /tmp/anno_v_rna_vec.bed' \
    '/Users/sethstadick/dev/biofast/bedcov/bedcov_c1_cgr /Users/sethstadick/Downloads/biofast-data-v1/ex-anno.bed /Users/sethstadick/Downloads/biofast-data-v1/ex-rna.bed > /tmp/c_anno_v_rna.bed'
md5sum /tmp/anno_v_rna.bed
md5sum /tmp/anno_v_rna_vec.bed
md5sum /tmp/c_anno_v_rna.bed

echo "===== g2r ====="
hyperfine \
    './coverage --method lapper_find_vectorized --bed-a /Users/sethstadick/Downloads/biofast-data-v1/ex-rna.bed --bed-b /Users/sethstadick/Downloads/biofast-data-v1/ex-anno.bed > /tmp/rna_v_anno.bed' \
    './coverage --method lapper_find --bed-a /Users/sethstadick/Downloads/biofast-data-v1/ex-rna.bed --bed-b /Users/sethstadick/Downloads/biofast-data-v1/ex-anno.bed > /tmp/rna_v_anno_vec.bed' \
    '/Users/sethstadick/dev/biofast/bedcov/bedcov_c1_cgr /Users/sethstadick/Downloads/biofast-data-v1/ex-rna.bed /Users/sethstadick/Downloads/biofast-data-v1/ex-anno.bed > /tmp/c_rna_v_anno.bed'
md5sum /tmp/rna_v_anno.bed
md5sum /tmp/rna_v_anno_vec.bed
md5sum /tmp/c_rna_v_anno.bed