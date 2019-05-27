%% create_models.pl
%% usage: Under project directory (containing this file, create.pl)
%% $ swipl
%% ?- ['create.pl']
%% ?- main(50) % create 50 models (under project/DNNs/)

:- use_module(library(http/json)).

get_output_size([InputHeight, InputWidth, _], CurrentLayer, [OutputHeight, OutputWidth, OutputDepth]) :-
  CurrentLayer.type == conv2d,
  OutChannels is CurrentLayer.out_channels,
  KernelHeight is CurrentLayer.kernel_height,
  KernelWidth is CurrentLayer.kernel_width,
  PaddingHeight is CurrentLayer.padding_height,
  PaddingWidth is CurrentLayer.padding_width,
  StrideHeight is CurrentLayer.stride_height,
  StrideWidth is CurrentLayer.stride_width,

  OutputHeight is floor((InputHeight - KernelHeight + 2 * PaddingHeight) / StrideHeight + 1),
  OutputWidth is floor((InputWidth - KernelWidth + 2 * PaddingWidth) / StrideWidth + 1),
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
% main(Conv1KernelHeight) :-
%   % between(20, 200, Conv1KernelHeight),
%   (repeat, random_between(20, 200, Conv1KernelHeight)),
%   % random_between(20, 200, Conv1KernelHeight),
%   Conv1 = conv1{type: conv2d, out_channels: 96, kernel_height: Conv1KernelHeight, kernel_width: 11, padding_height: 0, padding_width: 0, stride_height: 4, stride_width: 4},
%   Pool1 = pool1{type: maxpool2d, kernel_height: 3, kernel_width: 3, stride_height: 2, stride_width: 2},
%   Conv2 = conv2{type: conv2d, out_channels: 256, kernel_height: 5, kernel_width: 5, padding_height: 2, padding_width: 2, stride_height: 1, stride_width: 1},
%   Pool2 = pool2{type: maxpool2d, kernel_height: 3, kernel_width: 3, stride_height: 2, stride_width: 2},
%   Conv3 = conv3{type: conv2d, out_channels: 384, kernel_height: 3, kernel_width: 3, padding_height: 1, padding_width: 1, stride_height: 1, stride_width: 1},
%   Conv4 = conv4{type: conv2d, out_channels: 384, kernel_height: 3, kernel_width: 3, padding_height: 1, padding_width: 1, stride_height: 1, stride_width: 1},
%   Conv5 = conv5{type: conv2d, out_channels: 256, kernel_height: 3, kernel_width: 3, padding_height: 1, padding_width: 1, stride_height: 1, stride_width: 1},
%   Pool3 = pool3{type: maxpool2d, kernel_height: 3, kernel_width: 3, stride_height: 2, stride_width: 2},
%   AlexNet = [Conv1, Pool1, Conv2, Pool2, Conv3, Conv4, Conv5, Pool3],
%   is_network_legal([227, 227, 3], AlexNet).


% https://stackoverflow.com/a/41212369

% Find 5 solutions for Conv1KernelHeight such that
% main_2(Listy) :-
%   once(findnsols(5, Conv1KernelHeight, main(Conv1KernelHeight), Listy)).


% Random solutions: https://stackoverflow.com/a/41427112

% This works as a query.
% main_3(Listy) :-
%   findnsols(10, Customer, (repeat, random_between(18, 66, Age), random_member(Name, [jodie, tengyu, adiche, tomoyo, wolfgang]), Customer = customer{age: Age, name: Name}), Listy).


% main_4(Listy) :-
%   findall(Customer, (repeat, random_between(18, 21, Age), random_member(Name, [jodie, tengyu, adiche, tomoyo, wolfgang]), Customer = customer{age: Age, name: Name}), Listy).


find_alexnet(AlexNet, [InputHeight, InputWidth, InputChannels]) :-
  repeat, % So that random_between can run more than once

  %% Conv1 sizes
  random_between(20, 200, Conv1NumKernels),
  random_between(0, 3, Conv1PaddingHeight),
  random_between(0, 3, Conv1PaddingWidth),
  random_between(1, 5, Conv1StrideHeight),
  random_between(1, 5, Conv1StrideWidth),
  % Limit upperbound of kernel sizes to positive output (W - **F** + 2P >= 0, that is, F <= W + 2P)
  Conv1KernelHeightUpperBound is InputHeight + 2 * Conv1PaddingHeight,
  Conv1KernelWidthUpperBound is InputWidth + 2 * Conv1PaddingWidth,
  random_between(1, Conv1KernelHeightUpperBound, Conv1KernelHeight),
  random_between(1, Conv1KernelWidthUpperBound, Conv1KernelWidth),

  Conv1 = conv1{type: conv2d, out_channels: Conv1NumKernels, kernel_height: Conv1KernelHeight, kernel_width: Conv1KernelWidth, padding_height: Conv1PaddingHeight, padding_width: Conv1PaddingWidth, stride_height: Conv1StrideHeight, stride_width: Conv1StrideWidth},

  get_output_size([InputHeight, InputWidth, InputChannels], Conv1, [Pool1InputHeight, Pool1InputWidth, Pool1InputDepth]),


  %% Pool1 sizes
  random_between(1, 5, Pool1StrideHeight),
  random_between(1, 5, Pool1StrideWidth),
  % Limit upperbound of kernel sizes to positive output (W - **F** >= 0, that is, F <= W)
  random_between(1, Pool1InputHeight, Pool1KernelHeight),
  random_between(1, Pool1InputWidth, Pool1KernelWidth),

  Pool1 = pool1{type: maxpool2d, kernel_height: Pool1KernelHeight, kernel_width: Pool1KernelWidth, stride_height: Pool1StrideHeight, stride_width: Pool1StrideWidth},

  get_output_size([Pool1InputHeight, Pool1InputWidth, Pool1InputDepth], Pool1, [Conv2InputHeight, Conv2InputWidth, Conv2InputDepth]),

  %% Conv2 sizes
  random_between(20, 200, Conv2NumKernels),
  random_between(0, 3, Conv2PaddingHeight),
  random_between(0, 3, Conv2PaddingWidth),
  random_between(1, 5, Conv2StrideHeight),
  random_between(1, 5, Conv2StrideWidth),
  % Limit upperbound of kernel sizes to positive output (W - **F** + 2P >= 0, that is, F <= W + 2P)
  Conv2KernelHeightUpperBound is Conv2InputHeight + 2 * Conv2PaddingHeight,
  Conv2KernelWidthUpperBound is Conv2InputWidth + 2 * Conv2PaddingWidth,
  random_between(1, Conv2KernelHeightUpperBound, Conv2KernelHeight),
  random_between(1, Conv2KernelWidthUpperBound, Conv2KernelWidth),

  Conv2 = conv2{type: conv2d, out_channels: Conv2NumKernels, kernel_height: Conv2KernelHeight, kernel_width: Conv2KernelWidth, padding_height: Conv2PaddingHeight, padding_width: Conv2PaddingWidth, stride_height: Conv2StrideHeight, stride_width: Conv2StrideWidth},

  get_output_size([Conv2InputHeight, Conv2InputWidth, Conv2InputDepth], Conv2, [Pool2InputHeight, Pool2InputWidth, Pool2InputDepth]),


  %% Pool2 sizes
  random_between(1, 5, Pool2StrideHeight),
  random_between(1, 5, Pool2StrideWidth),
  % Limit upperbound of kernel sizes to positive output (W - **F** >= 0, that is, F <= W)
  random_between(1, Pool2InputHeight, Pool2KernelHeight),
  random_between(1, Pool2InputWidth, Pool2KernelWidth),

  Pool2 = pool2{type: maxpool2d, kernel_height: Pool2KernelHeight, kernel_width: Pool2KernelWidth, stride_height: Pool2StrideHeight, stride_width: Pool2StrideWidth},

  get_output_size([Pool2InputHeight, Pool2InputWidth, Pool2InputDepth], Pool2, [Conv3InputHeight, Conv3InputWidth, Conv3InputDepth]),


  %% Conv3 sizes
  random_between(20, 200, Conv3NumKernels),
  random_between(0, 3, Conv3PaddingHeight),
  random_between(0, 3, Conv3PaddingWidth),
  random_between(1, 5, Conv3StrideHeight),
  random_between(1, 5, Conv3StrideWidth),
  % Limit upperbound of kernel sizes to positive output (W - **F** + 2P >= 0, that is, F <= W + 2P)
  Conv3KernelHeightUpperBound is Conv3InputHeight + 2 * Conv3PaddingHeight,
  Conv3KernelWidthUpperBound is Conv3InputWidth + 2 * Conv3PaddingWidth,
  random_between(1, Conv3KernelHeightUpperBound, Conv3KernelHeight),
  random_between(1, Conv3KernelWidthUpperBound, Conv3KernelWidth),

  Conv3 = conv3{type: conv2d, out_channels: Conv3NumKernels, kernel_height: Conv3KernelHeight, kernel_width: Conv3KernelWidth, padding_height: Conv3PaddingHeight, padding_width: Conv3PaddingWidth, stride_height: Conv3StrideHeight, stride_width: Conv3StrideWidth},

  get_output_size([Conv3InputHeight, Conv3InputWidth, Conv3InputDepth], Conv3, [Conv4InputHeight, Conv4InputWidth, Conv4InputDepth]),


  %% Conv4 sizes
  random_between(20, 200, Conv4NumKernels),
  random_between(0, 3, Conv4PaddingHeight),
  random_between(0, 3, Conv4PaddingWidth),
  random_between(1, 5, Conv4StrideHeight),
  random_between(1, 5, Conv4StrideWidth),
  % Limit upperbound of kernel sizes to positive output (W - **F** + 2P >= 0, that is, F <= W + 2P)
  Conv4KernelHeightUpperBound is Conv4InputHeight + 2 * Conv4PaddingHeight,
  Conv4KernelWidthUpperBound is Conv4InputWidth + 2 * Conv4PaddingWidth,
  random_between(1, Conv4KernelHeightUpperBound, Conv4KernelHeight),
  random_between(1, Conv4KernelWidthUpperBound, Conv4KernelWidth),

  Conv4 = conv4{type: conv2d, out_channels: Conv4NumKernels, kernel_height: Conv4KernelHeight, kernel_width: Conv4KernelWidth, padding_height: Conv4PaddingHeight, padding_width: Conv4PaddingWidth, stride_height: Conv4StrideHeight, stride_width: Conv4StrideWidth},

  get_output_size([Conv4InputHeight, Conv4InputWidth, Conv4InputDepth], Conv4, [Conv5InputHeight, Conv5InputWidth, Conv5InputDepth]),


  % Conv5 sizes
  random_between(20, 200, Conv5NumKernels),
  random_between(0, 3, Conv5PaddingHeight),
  random_between(0, 3, Conv5PaddingWidth),
  random_between(1, 5, Conv5StrideHeight),
  random_between(1, 5, Conv5StrideWidth),
  % Limit upperbound of kernel sizes to positive output (W - **F** + 2P >= 0, that is, F <= W + 2P)
  Conv5KernelHeightUpperBound is Conv5InputHeight + 2 * Conv5PaddingHeight,
  Conv5KernelWidthUpperBound is Conv5InputWidth + 2 * Conv5PaddingWidth,
  random_between(1, Conv5KernelHeightUpperBound, Conv5KernelHeight),
  random_between(1, Conv5KernelWidthUpperBound, Conv5KernelWidth),

  Conv5 = conv5{type: conv2d, out_channels: Conv5NumKernels, kernel_height: Conv5KernelHeight, kernel_width: Conv5KernelWidth, padding_height: Conv5PaddingHeight, padding_width: Conv5PaddingWidth, stride_height: Conv5StrideHeight, stride_width: Conv5StrideWidth},

  get_output_size([Conv5InputHeight, Conv5InputWidth, Conv5InputDepth], Conv5, [Pool3InputHeight, Pool3InputWidth, _]),


  %% Pool3 sizes
  random_between(1, 5, Pool3StrideHeight),
  random_between(1, 5, Pool3StrideWidth),
  % Limit upperbound of kernel sizes to positive output (W - **F** >= 0, that is, F <= W)
  random_between(1, Pool3InputHeight, Pool3KernelHeight),
  random_between(1, Pool3InputWidth, Pool3KernelWidth),

  Pool3 = pool3{type: maxpool2d, kernel_height: Pool3KernelHeight, kernel_width: Pool3KernelWidth, stride_height: Pool3StrideHeight, stride_width: Pool3StrideWidth},



  AlexNet = [Conv1, Pool1, Conv2, Pool2, Conv3, Conv4, Conv5, Pool3],
  is_network_legal([InputHeight, InputWidth, InputChannels], AlexNet).


find_alexnets(AlexNets) :-
  once(findnsols(10, AlexNet, find_alexnet(AlexNet, [2, 65, 1]), AlexNets)).


% output_10 :-
%   find_alexnets(AlexNets), open('test2.json', write, Stream), json_write_dict(Stream, AlexNets), close(Stream).

write_model_to_file_per_k(Dict, Dirname, K) :-
  atomic_list_concat([Dirname, '/', 'k_', K], Kname),
  make_directory(Kname),
  atomic_list_concat([Kname, '/', 'model_params.json'], Fname),
  open(Fname, write, Stream),
  json_write_dict(Stream, Dict),
  close(Stream).

write_model_to_file(Dict) :-
  timestring(Timestring),
  atomic_list_concat(['DNNs/', 'alexnet_2_65_1', Timestring], Dirname),
  make_directory(Dirname),
  maplist(write_model_to_file_per_k(Dict, Dirname), ['3', '4', '5']).

timestring(Timestring) :-
  get_time(Timestamp), format_time(atom(Timestring), '%Y%m%d%H%M%S%f', Timestamp).

find_and_write_alexnet :-
  find_alexnet(AlexNet, [2, 65, 1]), write_model_to_file(AlexNet).

main(HowMany) :-
  once(findnsols(HowMany, [], find_and_write_alexnet, _)).
