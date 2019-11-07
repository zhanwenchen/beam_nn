%% create_models.pl
%% usage: Under project directory (containing this file, create.pl)
%% $ swipl
%% ?- ['create.pl']
%% ?- main(50) % create 50 models (under project/DNNs/)
%% v1.0: Implemented FCN
%% v1.1: Implemented FCN with upsample layer dictionaries.
%% v1.2: Fixed conv1d height (kernel, stride, padding) > 1 issue.
%% v1.3: 1. Conv1, Conv2 now allow stride = 2. Conv3, Conv4 still force stride=1; 2. (1.) stride change enables normal conv kernel sizes between 3 and 8.
%% v1.4: 1. Fixed conv2d stride height = 0 issue; 2. Narrowed conv2d stride height from [1, 3] to [1, 2]; 3. Narrowed conv2d padding height from [0, 3] to [0, 2].
%% v1.5: Narrowed kernel height to 1 (conv1d) and [2, 3] (conv2d).
%% v1.6.0: 1. Authoritative change to conv kernel widths: between [3, 15]. 2 is too small, and 15 is pretty big already. 2. Implemented get_dict_from_json_file to read in JSON Prolog dicts. 3. Refactored get_dict_from_json_file to utils.pl module. 4. Integrated hyperparam_ranges_fcn.json file to parameterize search ranges (partially).
%  v1.6.1 Remove default LeakyReLU after last layer because regression should not have nonlinear output activation. Also added batch norm.
%  v1.6.2 Narrowed learning rate to either 1e-04 or 1e-05.
%  v1.6.3 Reduced patience from 30 to 20 in order to churn out more models
%  v1.6.4 Use new dataset - reject-only point targets.
%  v1.6.5 Use new dataset - smooth-only point targets.
%  v1.6.6 Non-random everything except for kernel sizes.
%% TODO: combine conv1d and conv2d: it's just a matter of height=1.
%% BUG: Potential mixup between random_member and random_between.
:- use_module(library(http/json)).
:- use_module(utils, [get_dict_from_json_file/2, get_lower_upper_bounds_from_list/3]).
:- set_prolog_flag(verbose, silent).
:- initialization(main).


% For conv1d, set all heights to 1 so that we have a record.
get_output_size([InputHeight, InputWidth, _], CurrentLayer, [OutputHeight, OutputWidth, OutputDepth]) :-
  % Read variables from dictionary
  CurrentLayer.type == conv1d,
  OutChannels is CurrentLayer.out_channels,
  CurrentLayer.kernel_height == 1,
  KernelWidth is CurrentLayer.kernel_width,
  % writeln(CurrentLayer.padding_height),
  CurrentLayer.padding_height == 0,
  CurrentLayer.stride_height == 0,
  % writeln(CurrentLayer.kernel_height),
  % PaddingHeight is 1,
  PaddingWidth is CurrentLayer.padding_width,
  % StrideHeight is 1,
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
  KernelHeight is 1,
  KernelWidth is CurrentLayer.kernel_width,
  StrideHeight is 1,
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
  get_dict_from_json_file(ModelParamsRangesDict, 'hyperparam_ranges_fcn.json'),

  % Conv1 sizes
  % random_between(10, 50, Conv1NumKernels),
  Conv1NumKernels is 18,
  Conv1KernelHeight is 1,
  random_between(3, 65, Conv1KernelWidth),
  % Conv1KernelWidth // 2,
  % Conv1KernelWidth is 3,
  Conv1PaddingHeight is 0,
  Conv1PaddingWidthUpper is Conv1KernelWidth // 2,
  random_between(0, Conv1PaddingWidthUpper, Conv1PaddingWidth),
  % Conv1PaddingWidth is 1,
  % ((InputHeight = 1, Conv1PaddingHeight is 0) ; (InputHeight = 2, random_between(0, 2, Conv1PaddingHeight))),
  % random_between(0, 3, Conv1PaddingHeight),
  % random_between(0, 3, Conv1PaddingWidth),
  Conv1StrideHeight is 0,
  Conv1StrideWidth is 2,
  % ((InputHeight = 1, Conv1StrideHeight is 0) ; (InputHeight = 2, random_between(1, 2, Conv1StrideHeight))),
  % random_between(1, 5, Conv1StrideHeight),
  % random_member(Conv1StrideWidth, [1, 2]),
  % Conv1StrideWidth is 1, % So that we no longer need pooling.
  % Limit upperbound of kernel sizes to positive output (W - **F** + 2P >= 0, that is, F <= W + 2P)
  % Conv1KernelHeight is 1,
  % Conv1KernelWidth is 3,
  % ((InputHeight = 1, Conv1KernelHeight is 1) ; (InputHeight = 2, random_between(2, 3, Conv1KernelHeight))),
  % Conv1KernelHeightUpperBound is InputHeight + 2 * Conv1PaddingHeight,
  % random_between(1, Conv1KernelHeightUpperBound, Conv1KernelHeight),

  % Conv1KernelWidthRange is ModelParamsRangesDict.conv1_kernel_width,
  % random_member(Conv1KernelWidth, Conv1KernelWidthRange),
  % get_lower_upper_bounds_from_list(Conv1KernelWidthLower, Conv1KernelWidthUpper, ModelParamsRangesDict.conv1_kernel_width),
  % random_between(Conv1KernelWidthLower, Conv1KernelWidthUpper, Conv1KernelWidth),
  % random_member(Conv1KernelWidth, ModelParamsRangesDict.conv1_kernel_width),
  % random_between(2, 11, Conv1KernelWidth),
  % Conv1KernelWidthUpperBound is InputWidth + 2 * Conv1PaddingWidth,
  % random_between(1, Conv1KernelWidthUpperBound, Conv1KernelWidth),
  OldConv1 = conv1{name: conv1, in_channels: InputChannels, type: conv2d, out_channels: Conv1NumKernels, kernel_height: Conv1KernelHeight, kernel_width: Conv1KernelWidth, padding_height: Conv1PaddingHeight, padding_width: Conv1PaddingWidth, stride_height: Conv1StrideHeight, stride_width: Conv1StrideWidth},
  ((InputHeight = 1, Conv1 = OldConv1.put(type, conv1d)) ; (InputHeight = 2, Conv1 = OldConv1.put(type, conv2d))),
  get_output_size([InputHeight, InputWidth, InputChannels], Conv1, [Conv2InputHeight, Conv2InputWidth, Conv2InputDepth]),

  %% Conv2
  % random_between(Conv1NumKernels, 100, Conv2NumKernels),
  Conv2NumKernels is 91,
  Conv2KernelHeight is 1,
  Conv2KernelWidth = Conv1KernelWidth,
  % Conv2NumKernels > Conv1NumKernels,
  Conv2PaddingHeight is 0,
  % Conv2PaddingWidth is 3,
  Conv2PaddingWidthUpper is Conv2KernelWidth // 2,
  random_between(0, Conv2PaddingWidthUpper, Conv2PaddingWidth),
  % ((InputHeight = 1, Conv2PaddingHeight is 0) ; (InputHeight = 2, random_between(0, 2, Conv2PaddingHeight))),
  % random_between(0, 3, Conv2PaddingHeight),
  % random_between(0, 3, Conv2PaddingWidth),
  % ((InputHeight = 1, Conv2StrideHeight is 0) ; (InputHeight = 2, random_between(1, 2, Conv2StrideHeight))),
  % random_between(1, 5, Conv2StrideHeight),
  Conv2StrideHeight is 0,
  Conv2StrideWidth is 2,
  % random_member(Conv2StrideWidth, [1, 2]),
  % Conv2StrideWidth is 1, % To eliminate the need for pooling.
  % Limit upperbound of kernel sizes to positive output (W - **F** + 2P >= 0, that is, F <= W + 2P)

  % Conv2KernelWidth is 11,
  % ((InputHeight = 1, Conv2KernelHeight is 1) ; (InputHeight = 2, random_between(2, 3, Conv2KernelHeight))),
  % Conv2KernelHeightUpperBound is Conv2InputHeight + 2 * Conv2PaddingHeight,
  % random_between(1, Conv2KernelHeightUpperBound, Conv2KernelHeight),
  % write(Conv2KernelHeightUpperBound),
  % Conv2KernelWidthRange is ModelParamsRangesDict.conv2_kernel_width,
  % get_lower_upper_bounds_from_list(Conv2KernelWidthLower, Conv2KernelWidthUpper, ModelParamsRangesDict.conv2_kernel_width),
  % random_between(Conv2KernelWidthLower, Conv2KernelWidthUpper, Conv2KernelWidth),
  % random_member(Conv2KernelWidth, ModelParamsRangesDict.conv2_kernel_width),

  % random_member(Conv2KernelWidth, Conv2KernelWidthRange),
  % random_between(2, 11, Conv2KernelWidth),
  % Conv2KernelWidthUpperBound is Conv2InputWidth + 2 * Conv2PaddingWidth,
  % random_between(1, Conv2KernelWidthUpperBound, Conv2KernelWidth),
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
  % random_between(10, Conv2NumKernels, Conv3NumKernels),
  % Conv3NumKernels < Conv2NumKernels,
  Conv3NumKernels is 88,
  Conv3KernelHeight is 1,
  Conv3KernelWidth = Conv2KernelWidth,
  Conv3PaddingHeight is 0,
  Conv3PaddingWidthUpper is Conv3KernelWidth // 2,
  random_between(0, Conv3PaddingWidthUpper, Conv3PaddingWidth),
  % Conv3PaddingWidth is 3,
  % ((InputHeight = 1, Conv3PaddingHeight is 0) ; (InputHeight = 2, random_between(0, 2, Conv3PaddingHeight))),
  % random_between(0, 3, Conv3PaddingHeight),
  % random_between(0, 3, Conv3PaddingWidth),
  Conv3StrideHeight is 0,
  Conv3StrideWidth is 1, % To eliminate the need for pooling.
  % ((InputHeight = 1, Conv3StrideHeight is 0) ; (InputHeight = 2, random_between(1, 2, Conv3StrideHeight))),
  % random_between(1, 5, Conv3StrideHeight),
  % random_member(Conv3StrideWidth, [1, 2]),
  % Conv3KernelWidth is 5,

  % ((InputHeight = 1, Conv3KernelHeight is 1) ; (InputHeight = 2, random_between(2, 3, Conv3KernelHeight))),
  % Limit upperbound of kernel sizes to positive output (W - **F** + 2P >= 0, that is, F <= W + 2P)
  % Conv3KernelHeightUpperBound is Conv3InputHeight + 2 * Conv3PaddingHeight,
  % random_between(1, Conv3KernelHeightUpperBound, Conv3KernelHeight),
  % write(Conv3KernelHeightUpperBound),
  % get_lower_upper_bounds_from_list(Conv3KernelWidthLower, Conv3KernelWidthUpper, ModelParamsRangesDict.conv3_kernel_width),
  % random_between(Conv3KernelWidthLower, Conv3KernelWidthUpper, Conv3KernelWidth),
  % random_member(Conv3KernelWidth, ModelParamsRangesDict.conv3_kernel_width),
  % Conv3KernelWidthRange is ModelParamsRangesDict.conv3_kernel_width,
  % random_member(Conv3KernelWidth, Conv3KernelWidthRange),
  % random_between(2, 11, Conv3KernelWidth),
  % Conv3KernelWidthUpperBound is Conv3InputWidth + 2 * Conv3PaddingWidth,
  % random_between(1, Conv3KernelWidthUpperBound, Conv3KernelWidth),
  OldConv3 = conv3{name: conv3, in_channels: Conv2NumKernels, out_channels: Conv3NumKernels, kernel_height: Conv3KernelHeight, kernel_width: Conv3KernelWidth, padding_height: Conv3PaddingHeight, padding_width: Conv3PaddingWidth, stride_height: Conv3StrideHeight, stride_width: Conv3StrideWidth},
  ((InputHeight = 1, Conv3 = OldConv3.put(type, conv1d)) ; (InputHeight = 2, Conv3 = OldConv3.put(type, conv2d))),
  get_output_size([Conv3InputHeight, Conv3InputWidth, Conv3InputDepth], Conv3, [Upsample2InputHeight, Upsample2InputWidth, Upsample2InputDepth]),


  % Upsample2
  Upsample2 = upsample2{name: upsample2, type: upsample, scale_factor_height: 1, scale_factor_width: 2},
  get_output_size([Upsample2InputHeight, Upsample2InputWidth, Upsample2InputDepth], Upsample2, [Conv4InputHeight, Conv4InputWidth, Conv4InputDepth]),

  % Conv4
  Conv4NumKernels is InputChannels,
  Conv4KernelHeight is 1,
  Conv4KernelWidth = Conv3KernelWidth,
  Conv4PaddingHeight is 0,
  Conv4PaddingWidthUpper is Conv4KernelWidth // 2,
  random_between(0, Conv4PaddingWidthUpper, Conv4PaddingWidth),
  % Conv4PaddingWidth is 2,
  % ((InputHeight = 1, Conv4PaddingHeight is 0) ; (InputHeight = 2, random_between(0, 2, Conv4PaddingHeight))),
  % random_between(0, 3, Conv4PaddingHeight),
  % random_between(0, 3, Conv4PaddingWidth),
  % ((InputHeight = 1, Conv4StrideHeight is 0) ; (InputHeight = 2, random_between(1, 2, Conv4StrideHeight))),
  % random_between(1, 5, Conv4StrideHeight),
  Conv4StrideHeight is 0,
  Conv4StrideWidth is 1, % To eliminate the need for pooling.
  % ((InputHeight = 1, Conv4KernelHeight is 1) ; (InputHeight = 2, random_between(2, 3, Conv4KernelHeight))),
  % Limit upperbound of kernel sizes to positive output (W - **F** + 2P >= 0, that is, F <= W + 2P)
  % Conv4KernelHeightUpperBound is Conv4InputHeight + 2 * Conv4PaddingHeight,
  % random_between(1, Conv4KernelHeightUpperBound, Conv4KernelHeight),
  % write(Conv3KernelHeightUpperBound),
  % random_between(3, 8, Conv4KernelWidth),
  % Conv4KernelWidth is 3,
  % get_lower_upper_bounds_from_list(Conv4KernelWidthLower, Conv4KernelWidthUpper, ModelParamsRangesDict.conv4_kernel_width),
  % random_between(Conv4KernelWidthLower, Conv4KernelWidthUpper, Conv4KernelWidth),
  % random_member(Conv4KernelWidth, ModelParamsRangesDict.conv4_kernel_width),
  % Conv4KernelWidthRange is ModelParamsRangesDict.conv4_kernel_width,
  % random_member(Conv4KernelWidth, Conv4KernelWidthRange),
  % random_between(2, 11, Conv4KernelWidth),

  % Conv4KernelWidthUpperBound is Conv4InputWidth + 2 * Conv4PaddingWidth,
  % random_between(1, Conv4KernelWidthUpperBound, Conv4KernelWidth),
  OldConv4 = conv4{name: conv4, in_channels: Conv3NumKernels, out_channels: Conv4NumKernels, kernel_height: Conv4KernelHeight, kernel_width: Conv4KernelWidth, padding_height: Conv4PaddingHeight, padding_width: Conv4PaddingWidth, stride_height: Conv4StrideHeight, stride_width: Conv4StrideWidth},
  ((InputHeight = 1, Conv4 = OldConv4.put(type, conv1d)) ; (InputHeight = 2, Conv4 = OldConv4.put(type, conv2d))),
  get_output_size([Conv4InputHeight, Conv4InputWidth, Conv4InputDepth], Conv4, [InputHeight, InputWidth, InputChannels]),

  % TODO: implement is_network_legal for upsample dicts
  FCN = [Conv1, Conv2, Upsample1, Conv3, Upsample2, Conv4],
  % is_network_legal([InputHeight, InputWidth, InputChannels], FCN), writeln(Conv1NumKernels).
  is_network_legal([InputHeight, InputWidth, InputChannels], FCN), writeln(Conv4KernelWidth).

% model layers and training stuff
find_full_fcn(FCN) :-
  get_dict_from_json_file(ModelParamsRangesDict, 'hyperparam_ranges_fcn.json'),

  % random_member(InputDims, [[1, 130,1]]),
  InputDims = [1, 130, 1],
  % random_member(InputDims, [[1, 130,1], [1, 65, 2]]),
  % (InputDims = [2, 65, 1]; InputDims = [1, 130, 1]; InputDims = [1, 65, 2]),
  find_fcn(Layers, InputDims),

  LossFunction = 'MSE',
  % random_member(LossFunction, ['MSE', 'SmoothL1']),
  Optimizer = 'Adam',

  % LearningRateRange is ModelParamsRangesDict.learning_rate,
  LearningRate is 1E-5,
  % random_member(LearningRate, ModelParamsRangesDict.learning_rate),

  % LearningRate is 0.001,
  WeightDecay is 0,

  % Training and validation data locations
  NumScatter is 1,
  % random_between(1, 3, NumScatter),
  % TODO: switchable training data.
  DataDirname = 'data/20180402_L74_70mm',
  atomic_list_concat([DataDirname, '/train_', NumScatter, '.h5'], DataTrain),
  atomic_list_concat([DataDirname, '/val_', NumScatter, '.h5'], DataVal),

  Momentum is 0,
  writeln(InputDims),
  % writeln(NumScatter),

  Version = '1.6.6',

  FCN = model{model: 'FCN',
              input_dims: InputDims,
              version: Version,
              loss_function: LossFunction,
              optimizer: Optimizer,
              learning_rate: LearningRate,
              weight_decay: WeightDecay,
              data_is_target: 0,
              batch_size: 32,
              data_noise_gaussian: 1,
              patience: 20,
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
  Version = '1.6.6',
  timestring(Timestring),
  atomic_list_concat(['DNNs/', 'fcn_v', Version, '_', Timestring, '_created'], Dirname),
  make_directory(Dirname),
  maplist(write_model_to_file_per_k(Dict, Dirname), [3, 4, 5]),
  writeln(Dirname).

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
