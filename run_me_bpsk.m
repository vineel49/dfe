% decision feedback equalizer
% References: See Section 5.1.8 in the book "Digital Communications and
% Signal Processing" by K Vasudevan

clear all
close all
clc
training_len = 10^5; %length of the training sequence
snr_dB = 30; % snr in dB
ff_filter_len = 20; % feedforward filter length
fb_filter_len = 8; % feedback filter length
data_len = 10^6; % length of the data sequence
% snr parameters
snr = 10^(0.1*snr_dB);
noise_var = 1/(2*snr); % noise variance

% --------------- training phase ------------------------------------------
% source
training_a = randi([0 1],1,training_len);

% bpsk mapper
training_seq = 1-2*training_a;

% impulse response of the channel
fade_chan = [0.407 0.815 0.407]; % Proakis B channel
fade_chan = fade_chan/norm(fade_chan);
chan_len = length(fade_chan);

% awgn
noise = normrnd(0,sqrt(noise_var),1,training_len+chan_len-1);

% channel output
chan_op = conv(fade_chan,training_seq)+noise;

% ------------ LMS update of taps------------------------------------------
ff_filter = zeros(1,ff_filter_len); % feedforward filter initialization
fb_filter = zeros(1,fb_filter_len); % feedback filter initialization

ff_filter_ip = zeros(1,ff_filter_len); % feedforward filter input vector
fb_filter_ip = zeros(1,fb_filter_len); % feedback filter input vector

fb_filter_op = 0; % feedback filter output symbol

% estimating the autocorrelation of received sequence at zero lag
Rvv0 = (chan_op*chan_op')/(training_len+chan_len-1);

% maximum step size
max_step_size = 1/(ff_filter_len*Rvv0+fb_filter_len*(1));
step_size = 0.125*max_step_size; % step size

for i1=1:training_len-ff_filter_len+1 % steady state part
         ff_filter_ip(2:end)=ff_filter_ip(1:end-1);
         ff_filter_ip(1) = chan_op(i1);
         ff_filter_op = ff_filter*ff_filter_ip.'; % feedforward filter output
         
         ff_and_fb = ff_filter_op-fb_filter_op; 
         error = ff_and_fb-training_seq(i1); % instantaneous
         
         % hard decision
         temp = ff_and_fb<0;
         quantizer_op = 1-2*temp;
         
         % LMS update
         ff_filter=ff_filter-step_size*error*ff_filter_ip;
         fb_filter=fb_filter+step_size*error*fb_filter_ip;
         
         fb_filter_ip(2:end)=fb_filter(1:end-1);
         fb_filter_ip(1) = quantizer_op;
         
         fb_filter_op = fb_filter*fb_filter_ip.';
end

%-------    data transmission phase----------------------------
% source
data_a = randi([0 1],1,data_len);

% bpsk mapper
data_seq = 1-2*data_a;

% awgn
noise = normrnd(0,sqrt(noise_var),1,data_len+chan_len-1);

% channel output
chan_op = conv(fade_chan,data_seq)+noise;

dec_seq = zeros(1,training_len-ff_filter_len+1);% output from dfe
ff_filter_ip = zeros(1,ff_filter_len); % feedforward filter input
fb_filter_ip = zeros(1,fb_filter_len); % feedback filter input
fb_filter_op = 0; % feedback filter output symbol
for i1=1:training_len-ff_filter_len+1 % steady state part
         ff_filter_ip(2:end)=ff_filter_ip(1:end-1);
         ff_filter_ip(1) = chan_op(i1);
         ff_filter_op = ff_filter*ff_filter_ip.';
         
         ff_and_fb = ff_filter_op-fb_filter_op;
        
         % hard decision
         temp = ff_and_fb<0;
         dec_seq(i1) = 1-2*temp;
         
         fb_filter_ip(2:end)=fb_filter(1:end-1);
         fb_filter_ip(1) = dec_seq(i1);
         
         fb_filter_op = fb_filter*fb_filter_ip.'; % feedback filter output
end
% demapping symbols back to bits
dec_a = dec_seq<0;

% bit error rate
ber = nnz(dec_a-data_a(1:training_len-ff_filter_len+1))/(training_len-ff_filter_len+1)