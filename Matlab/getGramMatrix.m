function Q = getGramMatrix(samples)
	P = length(samples);

	Q = zeros(P, P);
	
	for i = 1 : P
		for j = 1 : P
			xi = samples(i).Item;
			xj = samples(j).Item;
			di = samples(i).Indicator;
			dj = samples(j).Indicator;
			
			Kij = xi' * xj + 1;
			
			Q(i, j) = di * dj * Kij;
		end
	end

end