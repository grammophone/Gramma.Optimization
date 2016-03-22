function lamda = svmTrain( samples, C )
	P = length(samples);

	[Q, g] = getGoalSystem(samples);
	
	'gram matrix'
	
	Q
	
	eigQ = eig(Q)
	
	lamda = quadprog(Q, g, [], [], [], [], zeros(P, 1), C/P*ones(P, 1));
end
