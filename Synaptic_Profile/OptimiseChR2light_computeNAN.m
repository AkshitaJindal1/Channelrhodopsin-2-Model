close all; clear; clc;

% Purpose: Find out what types of synaptic currents (parametrrised by alpha
% and beta) are actually achievable given the parameters of opsin. 

% Define the range of rise and decay values
rise_values = [0.01, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0];
decay_values = [1.0, 0.95, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35 ,0.3 ,0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025, 0.01];

% Initialize heatmap data and error matrices
heatmap_data = zeros(length(decay_values), length(rise_values));

addpath(genpath('OptimTraj'));

% Define the folder to save the figures
save_folder = 'figures';

% Check if the folder exists, if not create it
if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

% Loop over rise and decay values
for i = 1:length(decay_values)
    for j = 1:length(rise_values)
        % Set rise and decay values
        rise = rise_values(j);
        decay = decay_values(i);
        
        % Skip if rise or decay is zero to avoid division by zero
        if rise == 0 || decay == 0
            heatmap_data(i, j) = NaN;
            continue;
        end

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
        
        % Finding light input to generate synaptic dynamics according to four state model

        % Set target
        amp = -750;
        duration = 200;

        syn_signal = @(t) amp/(1/rise-1/decay)*(exp(-rise*t)-exp(-decay*t));

        % Plot target signal
        dt = 0.1;
        t = (0:dt:duration)';
        fig = figure('Name', sprintf('Figure_rise%.3f_decay%.3f', rise, decay));
        ax = axes(fig);
        hold(ax, 'on');
        p = plot(t, syn_signal(t), 'Linewidth', 2.0, 'HandleVisibility', 'off');
        color = get(p, 'Color');
        plot(t(1:10:end), syn_signal(t(1:10:end)), 'Linestyle', 'none', ...
          'Marker', '.', 'Markersize', 20, 'Color', color, ...
          'DisplayName', 'Target');
        xlabel(ax, 'Time (ms)');
        ylabel(ax, 'Current (pA)');
        set(ax, 'Fontsize', 20);
        legend(ax);
        drawnow;

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
        %                     Set up function handles                             %
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
        
        [modelpars, varpars, sysPars, fun] = setDefaultParameters_opsinmodel_ODE();

        % Hard coded parameters
        G = 6.4096082676895785; % ChR2 conductance
        gamma = 0.05; % fraction of current due to O2 state
        V = -80; % holding potential
        E = 0.0; % ChR2 reversal potential

        I_opsin = @(x) (G * (x(1,:) + gamma*x(3,:)).*(V - E));

        problem.func.dynamics = @(t,x,u) opsinmodel_ODE_ctrl(t, x, varpars, u);
        problem.func.pathObj = @(t,x,u) (I_opsin(x)-syn_signal(t)).^2; % minimise distance between target and actual

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
        %                     Find initial condition                              %
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
        
        init_cond = [0.0; 0.8; 0.1];
        x0 = fsolve(@(x) problem.func.dynamics(0,x,0), init_cond);

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
        %                     Set up problem bounds                               %
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
        problem.bounds.initialTime.low = 0;
        problem.bounds.initialTime.upp = 0;
        problem.bounds.finalTime.low = duration;
        problem.bounds.finalTime.upp = duration;

        problem.bounds.initialState.low = x0;
        problem.bounds.initialState.upp = x0;
        problem.bounds.finalState.low = zeros(3,1);
        problem.bounds.finalState.upp = ones(3,1);

        problem.bounds.state.low = zeros(3,1);
        problem.bounds.state.upp = ones(3,1);

        problem.bounds.control.low = 0.0;
        problem.bounds.control.upp = 1.0;

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
        %                    Initial guess at trajectory                          %
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
        problem.guess.time = [0,duration];
        problem.guess.state = repmat(x0, 1, 2);
        problem.guess.control = [0.1,0];  % Avoid zero control input

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
        %                         Solver options                                  %
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
        problem.options.method = 'hermiteSimpson';
        problem.options.nlpOpt = optimset(...
            'Display','off',...
            'MaxFunEvals',1e5); %uses IPOPT for iterations and hermite Simpson method for non-linear programming (NLP) 
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
        %                            Solve!                                       %
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

        try
            soln = optimTraj(problem);
        catch ME
            warning(['Optimization failed for rise = ', num2str(rise), ' and decay = ', num2str(decay)]);
            heatmap_data(i, j) = NaN;
            continue;
        end

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
        %                        Display Solution                                 %
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

        %% Unpack the simulation
        t_sol = linspace(soln.grid.time(1), soln.grid.time(end), 150);
        z = soln.interp.state(t_sol);
        u = soln.interp.control(t_sol);

        plot(t_sol, I_opsin(z), 'Linewidth', 2.0, ...
          'DisplayName', 'Photocurrent');
        yyaxis(ax, 'right');
        plot(ax, t_sol, 10*u, 'Linewidth', 2.0, 'HandleVisibility', 'off');
        ylabel(ax, '470 nm LED voltage (V)');

        %% Check solution
        [t,y] = ode45(@opsinmodel_ODE, [0,duration], x0, [], varpars, @(s) interp1(t_sol,u,s)) ;
        I = I_opsin(y');

        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
        %                     Define some solution measures                       %
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

        %% Peak current
        t_peak_target = log(decay/rise)/(decay - rise);
        peak_target_I = syn_signal(t_peak_target);
        t_init = linspace(0.0, 10.0, 10000);
        z_init = soln.interp.state(t_init);

        I_init = I_opsin(z_init);
        [peak_actual_I,ind] = min(I_init);

        peak_diff = peak_actual_I - peak_target_I;
        peak_percent_error = -100 * peak_diff / peak_target_I;
        
        % Peak latency
        t_peak_actual = t_init(ind);
        latency_diff = t_peak_actual - t_peak_target;
        latency_percent_error = 100 * latency_diff / t_peak_target;

        fprintf('Peak difference = %0.2f pA \n', peak_diff);
        fprintf('Peak percentage error = %0.2f%% \n', peak_percent_error);
        fprintf('Latency to peak difference = %0.2f ms \n', latency_diff);
        fprintf('Latency to peak percentage error = %0.2f%% \n', latency_percent_error);

        % Store the difference between actual and target peak
        heatmap_data(i, j) = peak_percent_error;

        % Save the figure
        fig_filename = fullfile(save_folder, sprintf('Figure_rise%.3f_decay%.3f.png', rise, decay));
        fprintf('Saving figure: %s\n', fig_filename); % Print out the filename
        saveas(fig, fig_filename);
        
    end
end

% Fill NaN values using interpolation
[X, Y] = meshgrid(rise_values, decay_values);
heatmap_data_filled = heatmap_data;
nan_mask = isnan(heatmap_data);
heatmap_data_filled(nan_mask) = griddata(X(~nan_mask), Y(~nan_mask), heatmap_data(~nan_mask), X(nan_mask), Y(nan_mask));

% Plot heatmap for peak percentage error
figure;

% Define the colormap
white_color = [1, 1, 1]; % White for NaNs

% Create heatmap
h = heatmap(rise_values, decay_values, heatmap_data_filled, 'Colormap', parula, 'MissingDataColor', white_color);

% Set color bar limits
h.ColorLimits = [0, 60];
h.MissingDataLabel = 'NaN';
h.GridVisible = 'off';

% Turn off the display of data labels
h.CellLabelColor = 'none';

xlabel('Rise');
ylabel('Decay');
title('Peak Percentage Error Heatmap');
