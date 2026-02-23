%% Stiff RLC Ladder: LGN-SD vs SymODEN
% 3 separate figures: Eigenvalues, Trajectory, Normalized Error
% Change 'datadir' to point to uniform/ or stiff/

clear; clc; close all;

%% ---- Config ----
datadir = 'exp5_stiff_6d/stiff';   % or 'exp5_stiff_6d/uniform'

%% ---- Load data ----
truth  = readmatrix('test_truth.csv');
pred_l = readmatrix( 'test_lgn.csv');
pred_s = readmatrix( 'test_sym.csv');
eig_t  = readmatrix('eig_true.csv');
eig_l  = readmatrix('eig_lgn.csv');
eig_s  = readmatrix('eig_sym.csv');
eig_det = readmatrix('eig_detail.csv');

t = truth(:,1);
x_true = truth(:, 2:end);
x_lgn  = pred_l(:, 2:end);
x_sym  = pred_s(:, 2:end);

%% ---- Compute errors ----
init_norm = sqrt(sum(x_true(1,:).^2));
abs_err_lgn = sqrt(sum((x_lgn - x_true).^2, 2));
abs_err_sym = sqrt(sum((x_sym - x_true).^2, 2));
true_norm = sqrt(sum(x_true.^2, 2));
eps = 1e-12 * init_norm;
nrel_lgn = abs_err_lgn ./ (true_norm + eps);
nrel_sym = abs_err_sym ./ (true_norm + eps);

nrmse_lgn = sqrt(mean(sum((x_lgn - x_true).^2, 2))) / init_norm;
nrmse_sym = sqrt(mean(sum((x_sym - x_true).^2, 2))) / init_norm;

fprintf('=== NRMSE ===\n');
fprintf('LGN-SD:  %.4e\n', nrmse_lgn);
fprintf('SymODEN: %.4e\n', nrmse_sym);
fprintf('Ratio:   %.1fx\n', nrmse_sym / nrmse_lgn);

%% ---- Colors ----
c_true = [0 0 0];
c_lgn  = [0 0 1];
c_sym  = [1 0 0];

%% Figure 1: Eigenvalues
figure(1);
set(gcf, 'Units', 'inches', 'Position', [1 1 6 5], 'Color', 'w');

scatter(eig_t(:,1), eig_t(:,2), 140, 'o', ...
    'MarkerEdgeColor', c_true, 'LineWidth', 2, ...
    'MarkerFaceColor', 'none', 'DisplayName', 'True'); hold on;
scatter(eig_l(:,1), eig_l(:,2), 55, '^', ...
    'MarkerFaceColor', c_lgn, 'MarkerEdgeColor', c_lgn, ...
    'DisplayName', 'LGN-SD');
scatter(eig_s(:,1), eig_s(:,2), 55, 's', ...
    'MarkerFaceColor', c_sym, 'MarkerEdgeColor', c_sym, ...
    'DisplayName', 'SymODEN');
xline(0, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, ...
    'HandleVisibility', 'off');

xlabel('Re(\lambda)'); ylabel('Im(\lambda)');
legend('Location', 'best'); legend boxoff;
box on; %grid on; set(gca, 'GridAlpha', 0.15);

% Annotate slow mode
[~, si] = max(eig_det(:,1));
text(eig_det(si,1)+0.01, eig_det(si,2)+0.15, ...
    sprintf('Slow mode\n\\tau \\approx %.0f', -1/eig_det(si,1)), ...
    'FontSize', 9, 'Color', [0.3 0.3 0.3]);

figor(1, "stiff_eigenvalues");

%% Figure 2: Trajectory x0(t)
figure(2);
set(gcf, 'Units', 'inches', 'Position', [1 1 6 5], 'Color', 'w');

plot(t, x_true(:,1), '-', 'Color', c_true, 'LineWidth', 2, ...
    'DisplayName', 'Ground Truth'); hold on;
plot(t, x_lgn(:,1), '--', 'Color', c_lgn, 'LineWidth', 1.5, ...
    'DisplayName', 'LGN-SD');
plot(t, x_sym(:,1), '--', 'Color', c_sym, 'LineWidth', 1.5, ...
    'DisplayName', 'SymODEN');

xlabel('Time (t)'); ylabel('Signal');
legend('Location', 'best'); legend boxoff;
box on; %grid on; set(gca, 'GridAlpha', 0.15);

xline(50, '--k', 'LineWidth', 1, 'HandleVisibility', 'off');
yl = ylim;
text(41, yl(2)*0.9, 'Train | Test', 'FontSize', 10, 'Color', [0.4 0.4 0.4]);
xlim([0, 120]);
figor(1, "stiff_trajectory");

%% Figure 3: Normalized Error
figure(3);
set(gcf, 'Units', 'inches', 'Position', [1 1 6 5], 'Color', 'w');

semilogy(t, nrel_lgn + 1e-16, '-', 'Color', c_lgn, 'LineWidth', 2, ...
    'DisplayName', 'LGN-SD'); hold on;
semilogy(t, nrel_sym + 1e-16, '-', 'Color', c_sym, 'LineWidth', 2, ...
    'DisplayName', sprintf('SymODEN', nrmse_sym));

xlabel('Time (t)');
ylabel('$$\frac{\|\hat{x} - x\|}{\|x\| + \epsilon}$$', 'Interpreter', 'latex');
legend('Location', 'best'); legend boxoff;
box on; %grid on; set(gca, 'GridAlpha', 0.15);

% text(t(end)*0.92, nrel_sym(end)*1.5, sprintf('%.1e', nrel_sym(end)), ...
%     'Color', c_sym, 'FontSize', 9, 'FontWeight', 'bold', 'HorizontalAlignment', 'right');
% text(t(end)*0.92, nrel_lgn(end)*0.6, sprintf('%.1e', nrel_lgn(end)), ...
%     'Color', c_lgn, 'FontSize', 9, 'FontWeight', 'bold', 'HorizontalAlignment', 'right');
xlim([0, 120]);
ylim([1e-6 1e-2])
figor(1, "stiff_normalized_error");

%% ---- Eigenvalue table ----
fprintf('\n=== EIGENVALUE TABLE ===\n');
fprintf('%-28s  %-12s  %-12s  %-10s\n', 'True λ', 'LGN err', 'Sym err', 'τ_mode');
fprintf('%s\n', repmat('-', 1, 68));
for i = 1:size(eig_det, 1)
    re = eig_det(i,1); im = eig_det(i,2);
    fprintf('%+8.5f %+8.5fj    %.4e    %.4e    %.1f\n', ...
        re, im, eig_det(i,5), eig_det(i,8), -1/re);
end
