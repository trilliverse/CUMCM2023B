% 读取 Excel 文件

% 读取所有数值数据（跳过前两行标题）
raw = readmatrix("E:\2025\math_model\CUMCM_Trian\20250801\CUMCM2023B\stats\附件.xlsx", 'FileType', 'spreadsheet', 'Range', 'B2:GS203');

% 提取横向坐标（由西向东）
x = raw(1, 2:end);  % 跳过第一列（纵向坐标）

% 提取纵向坐标（由南向北）
y = raw(2:end, 1);  % 跳过第一行（横向坐标）

% 提取 Z 值（海水深度）
Z = raw(2:end, 2:end);
Z = -1*Z

% 创建网格坐标
[X, Y] = meshgrid(x, y);

% 绘制 3D 曲面图
figure;
surf(X, Y, Z, 'EdgeColor', 'none');
colormap('parula');
colorbar;
xlabel('横向坐标 (NM)');
ylabel('纵向坐标 (NM)');
zlabel('海水深度 (m)');
title('海水深度 3D 曲面图');
grid on;
view(45, 30);

% 绘制等高线图
figure;
set(gcf, 'Color', 'white');  % 设置图形背景为白色
% 填充等高线图
contourf(X, Y, Z, 30, 'LineColor', 'none');
xlabel('横向坐标 (NM)');
ylabel('纵向坐标 (NM)');
title('海水深度 等高线图');

% 添加等高线标注
hold on;
% 指定需要标注的等高线深度值
label_levels = [-100,-80, -60, -50, -40,-30, -25, -20, -10, -5];  

% 添加图例
legend('海水深度', '标识等高线', 'Location', 'best');
hold off;