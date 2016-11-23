set terminal pngcairo font "arial,10" size 500,500
set output 'training and validation accuracy over steps.png'
set style fill solid
set boxwidth 1
set xlabel "Steps"
set ylabel "Accuracy"
plot "log" using 1:2 title "Training" with line linetype 2,\
 "log" using 1:3 title "Testing" with line linetype 3
