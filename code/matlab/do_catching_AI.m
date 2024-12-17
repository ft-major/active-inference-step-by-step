function [errors] = do_catching_AI(xc0, vb0, method, plot_flag)
% Unified script for "do_catching_AI_internalmodel" and "do_catching_AI_vectors_delay"
% Arguments:
% xc0 - Initial x-coordinate of the cursor
% vb0 - Initial velocity of the ball
% method - Specifies which implementation to use ('final_time_1g' , 'istxist_0g' , 'final_time_0g')
% plot_flag - Specifies whether the function will have a figure outcome

if nargin < 1, xc0 = 0; end
if nargin < 2, vb0 = 5; end
if nargin < 3, method = 'istxist_1g'; end
if nargin < 3, plot_flag = 1; end

rng(6);
% Simulation parameters
simTime = 1; %strcmp(method, 'final_time_1g') * 1 + strcmp(method, 'istxist_0g') * 1.5 + strcmp(method, 'final_time_0g') * 1;
dt = 0.001;
time = 0:dt:simTime;
N = length(time);
action = true;

% Action onset time
actionTime = (simTime / 4); %strcmp(method, 'final_time_1g') * (simTime / 4) + strcmp(method, 'istxist_0g') * 0 + strcmp(method, 'final_time_0g') * (simTime / 4);
deltaT = 0.1;

% Sensory variances
sigma_x = [0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.1, 0.1];
omega_x = 0.05;

% Generative process parameters
g = -9.81; k = 15; m = 0.05; c = sqrt(4 * k * m);
% g = -9.81; k = 5000; m = 5; c = sqrt(4 * k * m);

a(1) = 0;

% Initial generative process state
x(:, 1) = [0, vb0, 4.5, 0, 0, xc0, 0, 0];

% Dynamics equations
f_gp = @(x, a) [x(2), 0, x(4), g, 0, x(7), -(c / m) * x(7) + a, 0]';
f_gm = @(x) [x(2), 0, x(4), x(5), 0, x(7), x(8), 0]';

% Initialize variables
mu(:, 1) = x(:, 1);
rho(:, 1) = x(:, 1);

% Sensory noise
zgp_x(1,:) = randn (1 ,N)*.1 ;
zgp_x(2,:)  = randn (1 ,N)*.1 ;
zgp_x(3,:)  = randn (1 ,N)*.1 ;
zgp_x(4,:) = randn (1 ,N)*.1 ;
zgp_x(5,:) = zeros (1 ,N)*.1 ; % no sensory inputs on accelerations
zgp_x(6,:) = randn (1 ,N)*.1 ;
zgp_x(7,:) = randn (1 ,N)*.1 ;
zgp_x(8,:) = zeros (1 ,N)*.1 ; % no sensory inputs on accelerations

% Initialize error terms
epsilon_x(:, 1) = rho(:, 1) - mu(:, 1);
% xT = zeros(1, N);

% Method-specific deltaT calculation
% if strcmp(method, 'final_time_1g')
%     xT(1) = rho(1, 1) + deltaT * rho(2, 1);
% end

% Gradient descent learning parameters
ki_xb = [.1, .01];
ki_yb = [5, 5, .1];
ki_xc = [.0005, .005, .005];
ki = [ki_xb, ki_yb, ki_xc]';
ka = 50; %strcmp(method, 'final_time_1g') * 50 + strcmp(method, 'istxist_0g') * 40 + strcmp(method, 'final_time_0g') * 40;

% Simulation loop
for i = 2:N
    % Generative process (real world)
    x(:, i) = x(:, i - 1) + dt * (f_gp(x(:, i - 1), a(i - 1)));

    % Sensory inputs
    rho(:, i) = x(:, i) + zgp_x(:, i);

    % Sensory prediction errors
    epsilon_x(:, i) = rho(:, i - 1) - mu(:, i - 1);
    epsilon_x(5, i) = 0; % Ignore accelerations
    epsilon_x(8, i) = 0;

    % Method-specific deltaT and target calculation
    if strcmp(method, 'final_time_1g')
        deltaT(i) = 1 / g * (-rho(4, i - 1) - sqrt(rho(4, i - 1)^2 - 2 * g * rho(3, i - 1)));
        xT(i) = rho(1, i - 1) + deltaT(i) * rho(2, i - 1);
    elseif strcmp(method, 'istxist_0g')
        xT(i) = x(1, i - 1) + deltaT(i) * x(2, i - 1);
    elseif strcmp(method, 'final_time_0g')
        deltaT(i) = -rho(3, i - 1)/rho(4, i - 1);
        if isinf(deltaT(i))
            deltaT(i) = deltaT(i-1);
        end
        xT(i) = rho(1, i - 1) + deltaT(i) * rho(2, i - 1);

    elseif strcmp(method, 'istxist_1g')
        deltaY = 0.005;
deltaT(i) = 1 / g * (-rho(4, i - 1) - sqrt(rho(4, i - 1)^2 - 2 * g * deltaY));
        xT(i) = rho(1, i - 1) + deltaT(i) * rho(2, i - 1);
    end

    % Model errors
    epsilon_mu(:, i) = [0, 0, 0, mu(5, i - 1) - g, 0, 0, ...
        mu(8, i - 1) + c / m * mu(7, i - 1) + k / m * (mu(6, i - 1) - xT(i - 1)), 0];

    % Update states using gradient descent
    dFds(:, i) = -epsilon_x(:, i) ./ sigma_x';
    dFdmu(:, i) = [0; 0; 0; 0; epsilon_mu(4, i) / omega_x; ...
        epsilon_mu(7, i) * k / m / omega_x; epsilon_mu(7, i) * c / m / omega_x; epsilon_mu(7, i) / omega_x];
    mu(:, i) = mu(:, i - 1) + dt * (f_gm(mu(:, i - 1)) - ki .* (dFds(:, i) + dFdmu(:, i)));

    % Action update
    if time(i) > actionTime
        a(i) = a(i - 1) + dt * -ka * (1 / sigma_x(6) * epsilon_x(6, i) + 1 / sigma_x(7) * epsilon_x(7, i));
    else
        a(i) = 0;
    end
end

% plot

if plot_flag

    recordVideo = 0;
    figure ( 1 ) ; clf ;
    subplot (4 ,1 ,1)
    plot ( time , x(1,:) ,'k') ; hold on;
    plot ( time ,mu(1,:), ' b ' ) ; hold on;
    plot ( time , rho(1,:) , ' m ' ) ; hold on;
    legend ( ' xb ' , '  \mu_{xb} ' , ' \rho_{xb} ' )
    ylabel ( 'x_b ' )

    subplot (4 ,1 ,2)
    plot ( time , x(3,:),'k') ; hold on;
    plot ( time ,mu(3,:), 'b' ) ; hold on;
    plot ( time , rho(3,:) , 'm ') ; hold on;
    legend ( ' yb' , ' \mu_{yb}' , ' \rho_{yb}' ) ;
    ylabel ( 'y_b ' )


    subplot (4 ,1 ,3)
    plot ( time , x(2,:) ,'k') ; hold on;
    plot ( time ,mu(2,:), ' b ' ) ; hold on;
    plot ( time , rho(2,:) , ' m ' ) ; hold on;
    legend ( ' Vxb ' , '  V\mu_{xb} ' , ' V\rho_{xb} ' )
    ylabel ( 'Vx_b ' )

    subplot (4 ,1 ,4)
    plot ( time , x(4,:) ,'k') ; hold on;
    plot ( time ,mu(4,:), 'b' ) ; hold on;
    plot ( time , rho(4,:) , 'm ') ; hold on;
    legend ( ' Vyb' , ' V\mu_{yb}' , ' V\rho_{yb}' ) ;
    ylabel ( 'Vy_b ' )
    xlabel ( ' time ' ) ;

    figure ( 2 ) ; clf ;
    subplot (4 ,1 ,1)
    plot ( time , x(6,:) ,'k') ; hold on;
    plot ( time ,mu(6,:), ' b ' ) ; hold on;
    plot ( time , rho(6,:) , 'm ') ; hold on;
    legend ( ' xc ' , ' \mu_{xc} ', ' \rho_{xc} ' ) ;
    ylabel ( 'x_c ' )

    subplot (4 ,1 ,2)
    plot ( time , x(7,:) ,'k') ; hold on;
    plot ( time ,mu(7,:), ' b ' ) ; hold on;
    plot ( time , rho(7,:) , 'm ') ; hold on;
    legend ( ' Vxc ' , ' V\mu_{xc} ', ' V\rho_{xc} ' ) ;
    ylabel ( 'Vx_c ' )


    subplot (4 ,1 ,3)
    plot ( time , a , ' k ' ) ;
    ylabel ( ' a ' )

    figure ( 4 ) ; clf;
set(gcf,'position',[100,100,1200,500])
hold on;

for i = 1:4:length(x(1,:))
    subplot (1 ,2 ,2)
    title('Generative Process')
    plot(x(1,i),x(3,i),'k.'), hold on;

    if recordVideo

        if i > length(x(1,:))/2
            plot(x(6,i),0,'ko'),hold on;
        end
    else
        plot(x(6,i),0,'ko'),hold on;

    end
    ylim([-1, 6])
    xlim([-1, 6])
    xlabel('x (a.u.)')
    ylabel('y (a.u.)')

    % pause(0.01)
    drawnow
    subplot (1 ,2 ,1)
    title('Generative Model')
    plot(mu(1,i),mu(3,i),'b.'),hold on;

    % drawnow
    if recordVideo

        if i > length(x(1,:))/2
            plot(mu(6,i),0,'bo'),hold on;
        end
    else
        plot(mu(6,i),0,'bo'),hold on;

    end
    drawnow
    ylim([-1, 6])
    xlim([-1, 6])
    xlabel('x (a.u.)')
    ylabel('y (a.u.)')

    % pause(0.01) %Pause and grab frame
    if recordVideo
        pause(0.01)
        frame = getframe(gcf); %get frame
        writeVideo(myVideo, frame);
    end

end



end

% Calculate errors
xb = [x(1, :); x(3, :)];
xc = [x(6, :); zeros(size(x(6, :)))];

dist_mat = dist(xb', xc);
[~, ind_end_ball] = min(abs(xb(2, :)));
ind_end_ball = ind_end_ball(1);
[~, ind_err_istxist] = min(diag(dist_mat));
[~, ind_abserr] = min(abs(xc(1, :) - xb(1, ind_end_ball)));

temporal_error = time(ind_end_ball) - time(ind_err_istxist);
temporal_error_abs = time(ind_end_ball) - time(ind_abserr);
spatial_error_abs = min(abs(xc(1, :) - xb(1, ind_end_ball)));
spatial_error = xc(1, ind_end_ball) - xb(1, ind_end_ball);

errors = [temporal_error, spatial_error, temporal_error_abs, spatial_error_abs];
end
