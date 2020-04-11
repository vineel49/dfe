% decision feedback equalizer
% References: See Section 5.1.8 in the book "Digital Communications and
% Signal Processing" by K Vasudevan
% QPSK modulation

clear all
close all
clc
training_len = 10^5; %length of the training sequence
snr_dB = 40; % snr in dB
ff_filter_len = 100; % feedforward filter length
fb_filter_len = 80; % feedback filter length
data_len = 10^6; % length of the data sequence
fade_var_1D = 0.5; % 1D fade variance
chan_len = 5; % number of channel taps

% snr parameters
snr = 10^(0.1*snr_dB);
noise_var_1D = 2*(2*fade_var_1D*chan_len)/(2*snr); % noise variance

% --------------- training phase ------------------------------------------
% source
training_a = randi([0 1],1,2*training_len);

% qpsk mapper
training_seq = 1-2*training_a(1:2:end) + 1i*(1-2*training_a(2:2:end));

% impulse response of the channel
fade_chan = normrnd(0,sqrt(fade_var_1D),1,chan_len)+normrnd(0,sqrt(fade_var_1D),1,chan_len);
fade_chan = fade_chan/norm(fade_chan);
chan_len = length(fade_chan);

% awgn
noise = normrnd(0,sqrt(noise_var_1D),1,training_len+chan_len-1)+normrnd(0,sqrt(noise_var_1D),1,training_len+chan_len-1);

% channel output
chan_op = conv(fade_chan,training_seq)+noise;

% ------------ LMS update of taps------------------------------------------
ff_filter = zeros(1,ff_filter_len); % feedforward filter initialization
fb_filter = zeros(1,fb_filter_len); % feedback filter initialization

ff_filter_ip = zeros(1,ff_filter_len); % feedforward filter input vector
fb_filter_ip = zeros(1,fb_filter_len); % feedback filter input vector

fb_filter_op = 0; % feedback filter output symbol
% maximum step size
max_step_size = 1/(ff_filter_len*(2*2*fade_var_1D*chan_len+2*noise_var_1D)+fb_filter_len*(22));
step_size = 0.125*max_step_size; % step size
for i1=1:training_len-ff_filter_len+1 % steady state part
         ff_filter_ip(2:end)=ff_filter_ip(1:end-1);
         ff_filter_ip(1) = chan_op(i1);
         ff_filter_op = ff_filter*ff_filter_ip.'; % feedforward filter output
         
         ff_and_fb = ff_filter_op-fb_filter_op; 
         error = ff_and_fb-training_seq(i1); % instantaneous
         
         % hard decision
         temp1 = real(ff_and_fb)<0;
         temp2 = imag(ff_and_fb)<0;
         quantizer_op = 1-2*temp1 + 1i*(1-2*temp2);
         
         % LMS update
         ff_filter=ff_filter-step_size*error*conj(ff_filter_ip);
         fb_filter=fb_filter+step_size*error*conj(fb_filter_ip);
         
         fb_filter_ip(2:end)=fb_filter(1:end-1);
         fb_filter_ip(1) = quantizer_op;
         
         fb_filter_op = fb_filter*fb_filter_ip.';
end

%-------    data transmission phase----------------------------
% source
data_a = randi([0 1],1,2*data_len);

% qpsk mapper
data_seq = 1-2*data_a(1:2:end)+1i*(1-2*data_a(2:2:end));

% awgn
noise = normrnd(0,sqrt(noise_var_1D),1,data_len+chan_len-1)+...
    1i*normrnd(0,sqrt(noise_var_1D),1,data_len+chan_len-1);

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
         temp1 = real(ff_and_fb)<0;
         temp2 = imag(ff_and_fb)<0;
         dec_seq(i1) = 1-2*temp1 +1i*(1-2*temp2);
         
         fb_filter_ip(2:end)=fb_filter(1:end-1);
         fb_filter_ip(1) = dec_seq(i1);
         
         fb_filter_op = fb_filter*fb_filter_ip.'; % feedback filter output
end
% demapping symbols back to bits
dec_a = zeros(1,2*(training_len-ff_filter_len+1));
dec_a(1:2:end) = real(dec_seq)<0;
dec_a(2:2:end) = imag(dec_seq)<0;

% bit error rate
ber = nnz(dec_a-data_a(1:2*(training_len-ff_filter_len+1)))/(2*(training_len-ff_filter_len+1))