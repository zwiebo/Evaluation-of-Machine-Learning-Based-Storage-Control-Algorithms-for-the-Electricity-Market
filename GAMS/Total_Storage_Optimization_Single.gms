$ontext
####################
storage optimization
####################
 This model solve the optimal storage for X years
$offtext

Option threads=-1;
Option Reslim=10000;


         sets t/t0*t52583/
*         sets t/t1*t24/
;
         parameter        price /
$include C:\Python\Masterarbeit2.0\GAMS\GAMS_import_1d_2010-2015.txt
*$include test_t24.txt
/;

*Variables to define the properties of the storage technology
*        stor_eff                ..the loss during the feed in AND feed out
*                                  process
*        selfdischarge           ..percentual loss due to self discharge
*        max_charge_speed        ..maximum charge speed in dependency of the
*                                  installed capacity

scalar   stor_eff,
         selfdischarge,
         max_feed_speed;
         stor_eff = 0.825;
         selfdischarge = 0.007;
         max_feed_speed = 0.25;

positive variable
         feed_in(t),
         feed_out(t),
         stor_lvl(t);
                 feed_in.up(t)   = max_feed_speed;
                 feed_out.up(t)  = max_feed_speed;
                 stor_lvl.up(t)     = 1

variables
         hourly_profit(t),
         profit_acc(t);

free variable
         balance;

equations
         stor_lvl_eq(t),
*only for results file
         accumulate_profit(t),
         calc_hourly_profit(t),
*objective function
         calculate_balance;

stor_lvl_eq(t)..         stor_lvl(t) =e=
                                 stor_lvl(t-1)
                                 *(1-selfdischarge/100)
                                 + feed_in(t)
                                 - feed_out(t);

calc_hourly_profit(t)..     hourly_profit(t) =e= feed_out(t)
                                                 * price(t)
                                                 -(feed_in(t)
                                                 * price(t)
                                                 /stor_eff);

accumulate_profit(t)..      profit_acc(t)    =e= profit_acc(t-1)
                                                 + hourly_profit(t);

calculate_balance..         balance          =e= sum(t,  feed_out(t)
                                                         * price(t))
                                                 -sum(t, feed_in(t)
                                                          * price(t)/stor_eff);


model storOpt /all/;
solve storOpt using LP maximize balance;

File output/
"results_GAMS_10-15_TOTAL_single.csv"

/;
File output1/
"results_GAMS_10-15_TOTAL_single_wo_INFO_.csv"
/;

output.nd=5;
output1.nd=5;
PUT output;

Put 'RUNTIME;',system.time";"/;
Put 'RUNDATE;',system.date";"/;
Put 'balance;',balance.l";"/;
Put 'stor_eff;',stor_eff";"/;
Put 'selfdischarge;',selfdischarge";"/;
Put 'max_feed_speed;',max_feed_speed";"/;
Put'Timesstep;Price;Storage_Level;Feed_in;Feed_out;Balance;'/;

Loop(t,PUT  t.TL";"price(t)";" stor_lvl.l(t)";"feed_in.l(t)";"feed_out.l(t)";"hourly_profit.l(t)";"profit_acc.l(t)";"/)
putclose

put output1;
Put "price;Storage_Level;Feed_in;Feed_out;hourly_profit;profit_acc;"/;
Loop(t,PUT price(t)";"stor_lvl.l(t)";"feed_in.l(t)";"feed_out.l(t)";"hourly_profit.l(t)";"profit_acc.l(t)";"/)
putclose
parameter lc;
lc = sum(t,feed_in.l(t))
display lc
