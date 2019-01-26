function plot_digits(digit_matrix)
% plot_digits: Visualizes each example contained in digit_matrix.
%
% Note: N is the number of examples 
%       and M is the number of features per example.
%
% Inputs:
%   digits: N x M matrix of pixel intensities.
%
% Note: After calling this function, re-size your figure window so that
%       each pixel is approximately square.

CLASS_EXAMPLES_PER_PANE = 5;   %每个窗口没类放5个

% assume two evenly split classes
examples_per_class = size(digit_matrix,1)/2;    %每类的example数
num_panes = ceil(examples_per_class/CLASS_EXAMPLES_PER_PANE);   %需要的窗口数

for pane = 1:num_panes
  fprintf('Displaying pane %d/%d\n', pane, num_panes);

  % set up plot
  current_figure = figure;   %总体图像预处理
  colormap('gray');
  clf;
  
  for class_index = 1:2
    for example_index = 1:CLASS_EXAMPLES_PER_PANE    %控制一个窗口中每个class输出5个
      if (pane-1)*CLASS_EXAMPLES_PER_PANE + example_index > examples_per_class   %某个class的examples全部输完了
        break
      end
      
      % select appropriate subplot    %决定小图像在哪个位置
      digit_index = (class_index-1)*examples_per_class + ...
                    (pane-1)*CLASS_EXAMPLES_PER_PANE + example_index;   %现在是在digit_matrix的第几组数据
      
      subplot(2, CLASS_EXAMPLES_PER_PANE, ...
              (class_index-1)*CLASS_EXAMPLES_PER_PANE + example_index);
      
      % plot it
      current_pixels = reshape(digit_matrix(digit_index,:), 28, 28)';   %把现在的数据进行图像变换
      imagesc(current_pixels);   %按数值大小作图
      axis off;
    end
  end
  waitfor(current_figure);
end
