function fftPeakPower = getfftPeakPower(x,fs)

% This function takes n dimensional acceleration (and the sampling
% frequency of the accelerometer) as input and outputs the highest value of
% the magnitude squared of the Fourier Transform. It first filters the
% signal at 10 Hz (is this a good value, given that proper fleeing can go
% up to ~7 Hz?). Then, it pads the signal with zeros on the left and right
% (length of padding on each side corresponds to 1s if window length is 2
% seconds. Might later consider different/no padding if the signal is quite
% long, for instance, 10s long or so) and applies a Blackman-Harris window
% of length equal to that of the left-right padded signal. This is done to
% each of the three channels of the accelerometer. The Fourier transform is
% then computed (with a frequency precision of 0.01 Hz) for each channel.
% The average of the squared magnitudes of the three Fourier transforms is
% then computed. Finally, the peak of this averaged FFT is given as output.
% Note that the input can be 1D as well.
%
% These signal processing techniques were suggested by Hooman on 7th April,
% 2017.
%
% Written:      20th April, 2017
%               Pritish Chakravarty
%               LMAM, EPFL
%               [This function was written in the Kalahari]

D = size(x,2); % no. of dimensions of the input data


%% Normalizing the data from each channel of the accelerometer

% transforming data from each axis so that it has zero mean and
% sum-of-squares value equal to 1 (removing the DC component and
% normalizing the power of the signal)

m = mean(x);

if D>1
    m = repmat(m,size(x,1),1);
end

x = (x - m)./(sqrt(sum((x - m).^2)));


%% Filtering the signal at 10 Hz (?)

cutOff = 10; % lowpass cutoff frequency in Hertz
[b,a] = butter(4,cutOff/(fs/2),'low');
facc = filtfilt(b,a,x); % filtered three-dimensional acceleration


%% Windowing

lrfacc = [zeros(round(fs*1),D); facc; zeros(round(fs*1),D)];
    % padding the filtered signal with 1s of zeros on the left and right
    
w = window(@blackmanharris,size(lrfacc,1));

wlrfacc = w.*lrfacc;
    % windowing the left-right padded filtered signal
    

%% Computing the averaged FFT and finding the peak power

% the number of terms in the FFT is chosen to ensure a frequency resolution
% of 0.001 Hz
freqPrecision = 0.01; % desired frequency precision, in Hertz
nFreq = round(fs/freqPrecision);
FFTD = fft(wlrfacc,nFreq);

f = (0:nFreq-1)*fs/nFreq; % the frequencies, in Hertz, at which "fftX" is computed

% calculating the squared magnitude of each Fourier coefficient
avgP = sum(abs(FFTD).^2,2)/D;

idx = f<=fs/2;
f = f(idx);
avgP = avgP(idx);

fftPeakPower = max(avgP);