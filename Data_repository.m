% ***************************** INTERCONNECTION TRAINING DATASET *************************************

%%% INFO: This is a MatLab file containing the relevant data used to prepare the paper called
%%%      'Detection of Interconnections Between Unstable Resonant Orbits Via Machine Learning' 

%% DATASET (READY FOR TRAINING)

% A slight preprocessing is applied to the original 948 input features, to alleviate class 
% imbalance, eliminating for each classifier those that belong to the resonant family 
% (or class) it represents (see Subsection 4.2 in the paper). The training dataset is:
load("Datasets.mat");                    

% cols. 1-3: triplet {x,x_dot,C} of all existing resonant crossings at the energy range C = [2.955865333361881 3.024825648726140]
% col. 4: binary output indicating existence of interconnection (1) or not (0)
% col. 5: resonant class (family) to which the orbit crossing belongs (numbered 1 to 15)

%% INDEPENDENT TEST SET FOR VALIDATION (SUBSECTION V.A OF THE PAPER)

% Same cell structure as the 'Datasets' variable, but with a significantly 
% reduced number of rows in each classifier:
load("Testsets.mat");                    

%% MODELS DESIGNED FOR THE MULTI-CLASSIFIER BASED ON DNNs + BO

% A brief code is provided that loads the trained DNN models and processes the independent 
% test set for performance evaluation:

for i=1:length(Testsets)
    test = Testsets{i};
    % (DNNs + BO) trained classifiers: ---------------------------------------------------------------------
    if i == 1
       load('Trained_model_C1.mat')
    elseif i == 2
       load('Trained_model_C2.mat')
    elseif i == 4
       load('Trained_model_C4.mat')
    elseif i == 6
       load('Trained_model_C6.mat')
    elseif i == 8
       load('Trained_model_C8.mat')
    end
    Y_pred = classify(trainedNet,test(:,1:end-2));
    Y_test = test(:,4);
    TP_test = 0;FP_test = 0;TN_test = 0;FN_test = 0;
    for j=1:length(test(:,4))
        if Y_pred(j) == '1' && Y_test(j) == 1
           TP_test=TP_test+1;
        elseif Y_pred(j) == '1' && Y_test(j) == 0
           FP_test=FP_test+1;
        elseif Y_pred(j) == '0' && Y_test(j) == 0
           TN_test=TN_test+1;
        elseif Y_pred(j) == '0' && Y_test(j) == 1
           FN_test=FN_test+1;
        end
    end
    P_test = TP_test/(TP_test+FP_test); % Precision
    R_test = TP_test/(TP_test+FN_test); % Recall
    % Compute the F-score prioritising k times precision over recall:
    k = 5; % This is a value determined by experience. Since recall is also important, maybe this is enough.
    fk_score_test = (1+k^2)*P_test*R_test/(P_test+k^2*R_test); % F-k score
    % Define the objective function as the F-score error commited:
    fk_scoreError_test = 1-fk_score_test;
    accuracy = (TP_test+TN_test)/(TP_test+TN_test+FP_test+FN_test);
end
% NOTE: The information from the BO is provided in the corresponding
% variables. However, they are not needed for the performance evaluation.

%% MODELS DESIGNED FOR THE MULTI-CLASSIFIER BASED ON ELM/WELM + GA
