%% create_models.pl
%%
%% Usage 1: Under project directory (containing this file, create.pl)
%% $ swipl
%% ?- ['create.pl']
%% ?- main(50) % create 50 models (under project/DNNs/)
%%
%% Usage 2: Under project directory (containing this file, create.pl)
%% $ swipl create.pl 50 % create 50 models
%%
%% v1.0: Implemented FCN
%% v1.1: Implemented FCN with upsample layer dict. Use main/command-line args.
%% v1.2: Fixed conv1d kernel, padding, stride height > 1 issue.
%% TODO: combine conv1d and conv2d: it's just a matter of height=1.

:- use_module(library(http/json)).
:- set_prolog_flag(verbose, silent).
:- initialization(main).

% For conv1d, set all heights to 1 so that we have a record.
get_output_size([InputHeight, InputWidth, _], CurrentLayer, [OutputHeight, OutputWidth, OutputDepth]) :-
  % Read variables from dictionary
  CurrentLayer.type == conv1d,
  OutChannels is CurrentLayer.out_channels,
  KernelHeight is CurrentLayer.kernel_height,
  KernelHeight == 1,
  KernelWidth is CurrentLayer.kernel_width,
  PaddingHeight is CurrentLayer.padding_height,
  PaddingHeight == 1,
  PaddingWidth is CurrentLayer.padding_width,
  StrideHeight is CurrentLayer.stride_height,
  StrideHeight == 1,
  StrideWidth is CurrentLayer.stride_width,

  OutputHeight is 1,
  OutputWidth is floor((InputWidth - KernelWidth + 2 * PaddingWidth) / StrideWidth + 1),
  OutputDepth is OutChannels.

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

get_output_size([InputHeight, InputWidth, InputDepth], CurrentLayer, [OutputHeight, OutputWidth, OutputDepth]) :-
  CurrentLayer.type == upsample,
  ScaleFactorHeight is CurrentLayer.scale_factor_height,
  ScaleFactorWidth is CurrentLayer.scale_factor_width,

  OutputHeight is ScaleFactorHeight * InputHeight,
  OutputWidth is ScaleFactorWidth * InputWidth,
  OutputDepth is InputDepth.

get_output_size([InputHeight, InputWidth, InChannels], CurrentLayer, [OutputHeight, OutputWidth, OutputDepth]) :-
  CurrentLayer.type == maxpool1d,
  KernelHeight is CurrentLayer.kernel_height,
  KernelHeight == 1,
  KernelWidth is CurrentLayer.kernel_width,
  StrideHeight is CurrentLayer.stride_height,
  StrideHeight == 1,
  StrideWidth is CurrentLayer.stride_width,

  OutputHeight is 1,
  OutputWidth is floor((InputWidth - KernelWidth) / StrideWidth + 1),
  OutputDepth is InChannels.

get_output_size([InputHeight, InputWidth, InChannels], CurrentLayer, [OutputHeight, OutputWidth, OutputDepth]) :-
  CurrentLayer.type == maxpool2d,
  KernelHeight is CurrentLayer.kernel_height,
  KernelWidth is CurrentLayer.kernel_width,
  StrideHeight is CurrentLayer.stride_height,
  StrideWidth is CurrentLayer.stride_width,

  OutputHeight is floor((InputHeight - KernelHeight) / StrideHeight + 1),
  OutputWidth is floor((InputWidth - KernelWidth) / StrideWidth + 1),
  OutputDepth is InChannels.


get_output_size([_, _, InChannels], CurrentLayer, [OutputHeight, OutputWidth, OutputDepth]) :-
  CurrentLayer.type == adaptiveavgpool2d,

  OutputHeight is CurrentLayer.out_height,
  OutputWidth is CurrentLayer.out_width,
  OutputDepth is InChannels.

is_network_legal(_, []).
is_network_legal(_, [CurrentLayer]) :-
  CurrentLayer.type == fcs.
is_network_legal(InputSizes, [CurrentLayer|RestLayers]) :-
  get_output_size(InputSizes, CurrentLayer, [OutputHeight, OutputWidth, OutputDepth]),
  OutputHeight >= 1,
  OutputWidth >= 1,
  integer(OutputDepth), OutputDepth >= 1,
  is_network_legal([OutputHeight, OutputWidth, OutputDepth], RestLayers).

% Predicate to find fully-convolutional networks
find_fcn(FCN, [InputHeight, InputWidth, InputChannels]) :-
  % repeat, % So that random_between can run more than once.

  % Conv1 sizes
  random_between(20, 100, Conv1NumKernels),
  random_between(0, 2, Conv1PaddingHeight),
  random_between(0, 2, Conv1PaddingWidth),
  random_between(1, 2, Conv1StrideHeight),
  Conv1StrideWidth is 1, % So that we no longer need pooling.
  % Limit upperbound of kernel sizes to positive output (W - **F** + 2P >= 0, that is, F <= W + 2P)
  Conv1KernelHeightUpperBound is InputHeight + 2 * Conv1PaddingHeight,
  Conv1KernelWidthUpperBound is InputWidth + 2 * Conv1PaddingWidth,
  random_between(1, Conv1KernelHeightUpperBound, Conv1KernelHeight),
  random_between(1, Conv1KernelWidthUpperBound, Conv1KernelWidth),
  OldConv1 = conv1{name: conv1, in_channels: InputChannels, type: conv2d, out_channels: Conv1NumKernels, kernel_height: Conv1KernelHeight, kernel_width: Conv1KernelWidth, padding_height: Conv1PaddingHeight, padding_width: Conv1PaddingWidth, stride_height: Conv1StrideHeight, stride_width: Conv1StrideWidth},
  ((InputHeight = 1, Conv1 = OldConv1.put(type, conv1d)) ; (InputHeight = 2, Conv1 = OldConv1.put(type, conv2d))),
  get_output_size([InputHeight, InputWidth, InputChannels], Conv1, [Conv2InputHeight, Conv2InputWidth, Conv2InputDepth]),

  %% Conv2
  random_between(20, 100, Conv2NumKernels),
  Conv2NumKernels > Conv1NumKernels,
  random_between(0, 2, Conv2PaddingHeight),
  random_between(0, 2, Conv2PaddingWidth),
  random_between(1, 2, Conv2StrideHeight),
  Conv2StrideWidth is 1, % To eliminate the need for pooling.
  % Limit upperbound of kernel sizes to positive output (W - **F** + 2P >= 0, that is, F <= W + 2P)
  Conv2KernelHeightUpperBound is Conv2InputHeight + 2 * Conv2PaddingHeight,
  Conv2KernelWidthUpperBound is Conv2InputWidth + 2 * Conv2PaddingWidth,
  % write(Conv2KernelHeightUpperBound),
  random_between(1, Conv2KernelHeightUpperBound, Conv2KernelHeight),
  random_between(1, Conv2KernelWidthUpperBound, Conv2KernelWidth),
  OldConv2 = conv2{name: conv2, in_channels: Conv1NumKernels, out_channels: Conv2NumKernels, kernel_height: Conv2KernelHeight, kernel_width: Conv2KernelWidth, padding_height: Conv2PaddingHeight, padding_width: Conv2PaddingWidth, stride_height: Conv2StrideHeight, stride_width: Conv2StrideWidth},
  ((InputHeight = 1, Conv2 = OldConv2.put(type, conv1d)) ; (InputHeight = 2, Conv2 = OldConv2.put(type, conv2d))),
  get_output_size([Conv2InputHeight, Conv2InputWidth, Conv2InputDepth], Conv2, [Upsample1InputHeight, Upsample1InputWidth, Upsample1InputDepth]),

  % Upsample 1
  Upsample1 = upsample1{name: upsample1, type: upsample, scale_factor_height: 1, scale_factor_width: 2},
  % Upsample1FactorWidth is 2,
  % Conv3InputHeight is Upsample1InputHeight,
  % Conv3InputWidth is Upsample1InputWidth * 2,
  % atomic_list_concat(['Conv3InputWidth = ', Conv3InputWidth, '\n'], PrintString),
  % write(PrintString),
  % Conv3InputDepth is Upsample1InputDepth,
  get_output_size([Upsample1InputHeight, Upsample1InputWidth, Upsample1InputDepth], Upsample1, [Conv3InputHeight, Conv3InputWidth, Conv3InputDepth]),

  % Conv3
  random_between(20, 100, Conv3NumKernels),
  Conv3NumKernels > Conv2NumKernels,
  random_between(0, 2, Conv3PaddingHeight),
  random_between(0, 2, Conv3PaddingWidth),
  random_between(1, 2, Conv3StrideHeight),
  Conv3StrideWidth is 1, % To eliminate the need for pooling.
  % Limit upperbound of kernel sizes to positive output (W - **F** + 2P >= 0, that is, F <= W + 2P)
  Conv3KernelHeightUpperBound is Conv3InputHeight + 2 * Conv3PaddingHeight,
  Conv3KernelWidthUpperBound is Conv3InputWidth + 2 * Conv3PaddingWidth,
  % write(Conv3KernelHeightUpperBound),
  random_between(1, Conv3KernelHeightUpperBound, Conv3KernelHeight),
  random_between(1, Conv3KernelWidthUpperBound, Conv3KernelWidth),
  OldConv3 = conv3{name: conv3, in_channels: Conv2NumKernels, out_channels: Conv3NumKernels, kernel_height: Conv3KernelHeight, kernel_width: Conv3KernelWidth, padding_height: Conv3PaddingHeight, padding_width: Conv3PaddingWidth, stride_height: Conv3StrideHeight, stride_width: Conv3StrideWidth},
  ((InputHeight = 1, Conv3 = OldConv3.put(type, conv1d)) ; (InputHeight = 2, Conv3 = OldConv3.put(type, conv2d))),
  get_output_size([Conv3InputHeight, Conv3InputWidth, Conv3InputDepth], Conv3, [Upsample2InputHeight, Upsample2InputWidth, Upsample2InputDepth]),


  % Upsample2
  % Upsample2FactorWidth is 2,
  % Conv4InputHeight is Upsample2InputHeight,
  % Conv4InputWidth is Upsample2InputWidth * 2,
  % Conv4InputDepth is Upsample2InputDepth,
  Upsample2 = upsample2{name: upsample2, type: upsample, scale_factor_height: 1, scale_factor_width: 2},
  get_output_size([Upsample2InputHeight, Upsample2InputWidth, Upsample2InputDepth], Upsample2, [Conv4InputHeight, Conv4InputWidth, Conv4InputDepth]),

  % Conv4
  Conv4NumKernels is InputChannels,
  random_between(0, 2, Conv4PaddingHeight),
  random_between(0, 2, Conv4PaddingWidth),
  random_between(1, 2, Conv4StrideHeight),
  Conv4StrideWidth is 1, % To eliminate the need for pooling.
  % Limit upperbound of kernel sizes to positive output (W - **F** + 2P >= 0, that is, F <= W + 2P)
  Conv4KernelHeightUpperBound is Conv4InputHeight + 2 * Conv4PaddingHeight,
  Conv4KernelWidthUpperBound is Conv4InputWidth + 2 * Conv4PaddingWidth,
  % write(Conv3KernelHeightUpperBound),
  random_between(1, Conv4KernelHeightUpperBound, Conv4KernelHeight),
  random_between(1, Conv4KernelWidthUpperBound, Conv4KernelWidth),
  OldConv4 = conv4{name: conv4, in_channels: Conv3NumKernels, out_channels: Conv4NumKernels, kernel_height: Conv4KernelHeight, kernel_width: Conv4KernelWidth, padding_height: Conv4PaddingHeight, padding_width: Conv4PaddingWidth, stride_height: Conv4StrideHeight, stride_width: Conv4StrideWidth},
  ((InputHeight = 1, Conv4 = OldConv4.put(type, conv1d)) ; (InputHeight = 2, Conv4 = OldConv4.put(type, conv2d))),
  get_output_size([Conv4InputHeight, Conv4InputWidth, Conv4InputDepth], Conv4, [InputHeight, InputWidth, InputChannels]),

  % TODO: implement is_network_legal for upsample dicts
  FCN = [Conv1, Conv2, Upsample1, Conv3, Upsample2, Conv4],
  % is_network_legal([InputHeight, InputWidth, InputChannels], FCN), writeln(Conv1NumKernels).
  is_network_legal([InputHeight, InputWidth, InputChannels], FCN).

% model layers and training stuff
find_full_fcn(FCN) :-
  random_member(InputDims, [[2, 65, 1], [1, 130,1], [1, 65, 2]]),
  % (InputDims = [2, 65, 1]; InputDims = [1, 130, 1]; InputDims = [1, 65, 2]),
  find_fcn(Layers, InputDims),

  random_member(LossFunction, ['MSE', 'SmoothL1']), % CHANGED: no longer use L1, which has proven ineffective.
  Optimizer = 'Adam',
  LearningRate is 0.001,
  WeightDecay is 0,

  % Training and validation data locations
  random_between(1, 3, NumScatter),
  % TODO: switchable training data.
  DataDirname = '/Users/zhanwenchen/Documents/projects/beam_nn/data/20180402_L74_70mm',
  atomic_list_concat([DataDirname, '/train_', NumScatter, '.h5'], DataTrain),
  atomic_list_concat([DataDirname, '/val_', NumScatter, '.h5'], DataVal),

  Momentum is 0,
  % writeln(InputDims),
  % writeln(NumScatter),

  FCN = model{model: 'FCN',
              input_dims: InputDims,
              version: '0.1',
              loss_function: LossFunction,
              optimizer: Optimizer,
              learning_rate: LearningRate,
              weight_decay: WeightDecay,
              data_is_target: 0,
              batch_size: 32,
              data_noise_gaussian: 1,
              patience: 50,
              data_train: DataTrain,
              data_val: DataVal,
              momentum: Momentum,
              layers: Layers}.

write_model_to_file_per_k(Dict, Dirname, K) :-
  atomic_list_concat([Dirname, '/', 'k_', K], Kname),
  make_directory(Kname),
  atomic_list_concat([Kname, '/', 'model_params.json'], Fname),
  NewDict = Dict.put(k, K),
  open(Fname, write, Stream),
  json_write_dict(Stream, NewDict),
  close(Stream).

write_model_to_file(Dict) :-
  timestring(Timestring),
  atomic_list_concat(['DNNs/', 'fcn_', Timestring, '_created'], Dirname),
  make_directory(Dirname),
  maplist(write_model_to_file_per_k(Dict, Dirname), [3, 4, 5]).
  % write(Dirname).

timestring(Timestring) :-
  get_time(Timestamp), format_time(atom(Timestring), '%Y%m%d%H%M%S%f', Timestamp).

find_and_write_fcn :-
  find_full_fcn(FCN), write_model_to_file(FCN).

main :-
  current_prolog_flag(argv, [HowManyStr|_]),
  atom_number(HowManyStr, HowManyInt),
  once(findnsols(HowManyInt, [], (repeat, find_and_write_fcn), Y)),
  halt.

% main(HowMany) :-
%   once(findnsols(HowMany, [], (repeat, find_and_write_fcn), Y)).
