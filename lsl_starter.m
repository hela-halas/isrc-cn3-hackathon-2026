% lsl_starter.m
%
% MATLAB equivalent of lsl_starter.py.
%
% This is intentionally minimal. It only:
%   1. Loads the LSL library and resolves an EEG stream.
%   2. Opens an inlet and validates the connection by pulling data.
%   3. Grabs one fixed-length window of raw samples.
%   4. Prints a preview and plots the window as a sanity check.
%
% Everything else -- filtering, spatial/feature extraction, classifier
% training or prediction, and any output (serial, LSL out, etc.) is left
% for you to build on top of this.
%
% Requires liblsl-Matlab to be on the path, e.g.:
%   addpath(genpath('D:\MEEG_LSL\'));

%% ---- settings you may want to change ----
LSL_type    = 'EEG';   % stream type to look for
numChn      = 8;       % how many channels to keep (FlexEEG: 8 EEG + 1 aux)
windowSecs  = 3;        % seconds of data to grab

%% ---- connect ----
disp('Loading the LSL library...');
lib = lsl_loadlib();

disp('Resolving an EEG stream...');
result = {};
while isempty(result)
    result = lsl_resolve_byprop(lib, 'type', LSL_type);
end

disp('Opening an inlet...');
inlet = lsl_inlet(result{1});

streamInfo = inlet.info();
fs = streamInfo.nominal_srate();
fprintf('Connected: %d channels at %g Hz\n', streamInfo.channel_count(), fs);

%% ---- validate the connection ----
% LSL sometimes takes a moment to establish a streaming connection, so
% pull data until something actually arrives before starting properly.
disp('Checking connection...');
while true
    [chunk, ~] = inlet.pull_chunk();
    if ~isempty(chunk)
        break;
    end
end

%% ---- grab one window ----
wantedSamples = round(windowSecs * fs);
buffer = [];
while size(buffer, 2) < wantedSamples
    [chunk, ~] = inlet.pull_chunk();
    if ~isempty(chunk)
        buffer = horzcat(buffer, chunk); %#ok<AGROW>
    end
end
epoch = buffer(1:numChn, end-wantedSamples+1:end);
fprintf('Collected epoch with size %dx%d (channels x samples) at %g Hz\n', ...
    size(epoch, 1), size(epoch, 2), fs);

%% ------------------------------------------------------------------
% SANITY-CHECK SCAFFOLDING -- delete this block once you trust that
% data is actually arriving and start building your own pipeline.
% ------------------------------------------------------------------
disp('First few raw samples per channel:');
for ch = 1:size(epoch, 1)
    fprintf('  ch%d: %s ...\n', ch - 1, mat2str(epoch(ch, 1:5), 4));
end

timeAxis = (0:size(epoch, 2) - 1) / fs;
figure;
plot(timeAxis, epoch');
xlabel('seconds');
ylabel('amplitude');
title('Raw EEG window (sanity check only)');
legendLabels = arrayfun(@(c) sprintf('ch%d', c), 0:size(epoch, 1) - 1, ...
    'UniformOutput', false);
legend(legendLabels, 'Location', 'northeastoutside');
% ------------------------------------------------------------------
% END SANITY-CHECK SCAFFOLDING
% ------------------------------------------------------------------

% TODO: this is where your own processing / classification pipeline starts.