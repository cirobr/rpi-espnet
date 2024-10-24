rm ht.csv
# rm ht.txt
rm out.txt
julia -t1 ht.jl 0 200 false > out.txt
