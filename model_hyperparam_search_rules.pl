get_output_size([InputHeight, InputWidth, _], CurrentLayer, [OutputHeight, OutputWidth, OutputDepth]) :-
  CurrentLayer.type == conv2d,
  OutChannels is CurrentLayer.out_channels,
  KernelHeight is CurrentLayer.kernel_height,
  KernelWidth is CurrentLayer.kernel_width,
  PaddingHeight is CurrentLayer.padding_height,
  PaddingWidth is CurrentLayer.padding_width,
  StrideHeight is CurrentLayer.stride_height,
  StrideWidth is CurrentLayer.stride_width,

  OutputHeight is (InputHeight - KernelHeight + 2 * PaddingHeight) / StrideHeight + 1,
  OutputWidth is (InputWidth - KernelWidth + 2 * PaddingWidth) / StrideWidth + 1,
  OutputDepth is OutChannels.


get_output_size([InputHeight, InputWidth, InChannels], CurrentLayer, [OutputHeight, OutputWidth, OutputDepth]) :-
  CurrentLayer.type == maxpool2d,
  KernelHeight is CurrentLayer.kernel_height,
  KernelWidth is CurrentLayer.kernel_width,
  StrideHeight is CurrentLayer.stride_height,
  StrideWidth is CurrentLayer.stride_width,

  OutputHeight is floor((InputHeight - KernelHeight) / StrideHeight + 1),
  OutputWidth is floor((InputWidth - KernelWidth) / StrideWidth + 1),
  OutputDepth is InChannels.


is_network_legal(_, []).
is_network_legal(InputSizes, [CurrentLayer|RestLayers]) :-
  get_output_size(InputSizes, CurrentLayer, [OutputHeight, OutputWidth, OutputDepth]),
  OutputHeight >= 1,
  OutputWidth >= 1,
  integer(OutputDepth), OutputDepth >= 1,
  is_network_legal([OutputHeight, OutputWidth, OutputDepth], RestLayers).

%% Conv1 = conv1{type: conv2d, out_channels: 96, kernel_height: 11, kernel_width: 11, padding_height: 0, padding_width: 0, stride_height: 4, stride_width: 4}, get_output_size([227, 227, 3], X, [OutputHeight, OutputWidth, OutputDepth]).

% main :-
%   Conv1 = conv1{type: conv2d, out_channels: 96, kernel_height: 11, kernel_width: 11, padding_height: 0, padding_width: 0, stride_height: 4, stride_width: 4},
%   Pool1 = pool1{type: maxpool2d, kernel_height: 3, kernel_width: 3, stride_height: 2, stride_width: 2},
%   Conv2 = conv2{type: conv2d, out_channels: 256, kernel_height: 5, kernel_width: 5, padding_height: 2, padding_width: 2, stride_height: 1, stride_width: 1},
%   Pool2 = pool2{type: maxpool2d, kernel_height: 3, kernel_width: 3, stride_height: 2, stride_width: 2},
%   Conv3 = conv3{type: conv2d, out_channels: 384, kernel_height: 3, kernel_width: 3, padding_height: 1, padding_width: 1, stride_height: 1, stride_width: 1},
%   Conv4 = conv4{type: conv2d, out_channels: 384, kernel_height: 3, kernel_width: 3, padding_height: 1, padding_width: 1, stride_height: 1, stride_width: 1},
%   Conv5 = conv5{type: conv2d, out_channels: 256, kernel_height: 3, kernel_width: 3, padding_height: 1, padding_width: 1, stride_height: 1, stride_width: 1},
%   Pool3 = pool3{type: maxpool2d, kernel_height: 3, kernel_width: 3, stride_height: 2, stride_width: 2},
%   AlexNet = [Conv1, Pool1, Conv2, Pool2, Conv3, Conv4, Conv5, Pool3],
%   is_network_legal([227, 227, 3], AlexNet).

% Use of repeat, random: https://stackoverflow.com/a/36350988
main(Conv1KernelHeight) :-
  % between(20, 200, Conv1KernelHeight),
  (repeat, random_between(20, 200, Conv1KernelHeight)),
  % random_between(20, 200, Conv1KernelHeight),
  Conv1 = conv1{type: conv2d, out_channels: 96, kernel_height: Conv1KernelHeight, kernel_width: 11, padding_height: 0, padding_width: 0, stride_height: 4, stride_width: 4},
  Pool1 = pool1{type: maxpool2d, kernel_height: 3, kernel_width: 3, stride_height: 2, stride_width: 2},
  Conv2 = conv2{type: conv2d, out_channels: 256, kernel_height: 5, kernel_width: 5, padding_height: 2, padding_width: 2, stride_height: 1, stride_width: 1},
  Pool2 = pool2{type: maxpool2d, kernel_height: 3, kernel_width: 3, stride_height: 2, stride_width: 2},
  Conv3 = conv3{type: conv2d, out_channels: 384, kernel_height: 3, kernel_width: 3, padding_height: 1, padding_width: 1, stride_height: 1, stride_width: 1},
  Conv4 = conv4{type: conv2d, out_channels: 384, kernel_height: 3, kernel_width: 3, padding_height: 1, padding_width: 1, stride_height: 1, stride_width: 1},
  Conv5 = conv5{type: conv2d, out_channels: 256, kernel_height: 3, kernel_width: 3, padding_height: 1, padding_width: 1, stride_height: 1, stride_width: 1},
  Pool3 = pool3{type: maxpool2d, kernel_height: 3, kernel_width: 3, stride_height: 2, stride_width: 2},
  AlexNet = [Conv1, Pool1, Conv2, Pool2, Conv3, Conv4, Conv5, Pool3],
  is_network_legal([227, 227, 3], AlexNet).


% https://stackoverflow.com/a/41212369

% Find 5 solutions for Conv1KernelHeight such that
main_2(Listy) :-
  once(findnsols(5, Conv1KernelHeight, main(Conv1KernelHeight), Listy)).


% Random solutions: https://stackoverflow.com/a/41427112

% This works as a query.
main_3(Listy) :-
  findnsols(10, Customer, (repeat, random_between(18, 66, Age), random_member(Name, [jodie, tengyu, adiche, tomoyo, wolfgang]), Customer = customer{age: Age, name: Name}), Listy).


% main_4(Listy) :-
%   findall(Customer, (repeat, random_between(18, 21, Age), random_member(Name, [jodie, tengyu, adiche, tomoyo, wolfgang]), Customer = customer{age: Age, name: Name}), Listy).


find_alexnet(AlexNet) :-
  repeat,
  random_between(20, 200, Conv1KernelHeight), random_between(20, 200, Conv1KernelWidth), random_between(0, 3, Conv1PaddingHeight), random_between(0, 3, Conv1PaddingWidth), random_between(1, 5, Conv1StrideHeight), random_between(1, 5, Conv1StrideWidth),
  Conv1 = conv1{type: conv2d, out_channels: 96, kernel_height: Conv1KernelHeight, kernel_width: Conv1KernelWidth, padding_height: Conv1PaddingHeight, padding_width: Conv1PaddingWidth, stride_height: Conv1StrideHeight, stride_width: Conv1StrideWidth},
  Pool1 = pool1{type: maxpool2d, kernel_height: 3, kernel_width: 3, stride_height: 2, stride_width: 2},
  Conv2 = conv2{type: conv2d, out_channels: 256, kernel_height: 5, kernel_width: 5, padding_height: 2, padding_width: 2, stride_height: 1, stride_width: 1},
  Pool2 = pool2{type: maxpool2d, kernel_height: 3, kernel_width: 3, stride_height: 2, stride_width: 2},
  Conv3 = conv3{type: conv2d, out_channels: 384, kernel_height: 3, kernel_width: 3, padding_height: 1, padding_width: 1, stride_height: 1, stride_width: 1},
  Conv4 = conv4{type: conv2d, out_channels: 384, kernel_height: 3, kernel_width: 3, padding_height: 1, padding_width: 1, stride_height: 1, stride_width: 1},
  Conv5 = conv5{type: conv2d, out_channels: 256, kernel_height: 3, kernel_width: 3, padding_height: 1, padding_width: 1, stride_height: 1, stride_width: 1},
  Pool3 = pool3{type: maxpool2d, kernel_height: 3, kernel_width: 3, stride_height: 2, stride_width: 2},
  AlexNet = [Conv1, Pool1, Conv2, Pool2, Conv3, Conv4, Conv5, Pool3],
  is_network_legal([227, 227, 3], AlexNet).


find_alexnets(AlexNets) :-
  once(findnsols(10, AlexNet, find_alexnet(AlexNet), AlexNets)).
