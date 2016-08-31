import urllib, json, pdb, math, numpy as np

def get_USD_exchange(curr):
	''' 
	@Desc: 	Loads the exchange rates of a currency to USD from yahoo.finance.historicaldata
	@Param: curr (String) ex: "EUR"
	@Returns sorted (ascending) list of dictionaries {'Volume': <val>, 'Open': <val>, 'Close': <val>, ...}
	'''
	qString = 'select * from yahoo.finance.historicaldata where symbol = "'+curr+'=X" and startDate = "2015-01-01" and endDate = "2016-01-01"';
	qString_encoded = urllib.quote_plus(qString)
	url = "https://query.yahooapis.com/v1/public/yql?q="+qString_encoded+"&format=json&diagnostics=true&env=store%3A%2F%2Fdatatables.org%2Falltableswithkeys&callback="
	response = urllib.urlopen(url)
	data = json.loads(response.read())
	if data['query']['results'] == None:
		return None
	else:
		ret = list(reversed(data['query']['results']['quote']))
		return ret

def get_currencies():
	''' 
	@Desc: 	Loads the currency name & abbrevation from local file 
	@Returns dictionary = {abbrevation: {name: <name>}, ...}
	'''
	f = open("waehrungen.csv")
	currencies = {}
	for line in f:
		split = line.replace("\r\n","").split(";")
		currencies[split[1]] = {"name": split[0]}
	return currencies

def get_USD_rates():
	''' 
	@Desc: 	Writes file with exchange rate matrix of currencies. This rate is the value in USD.
	@Returns void, local file "xchange_rates.csv" with content: <currency>\t<rate1>\t<rate2>...\t<rateN>\n
	'''
	f = open('usd_xchange.txt', 'w+')
	currencies = get_currencies()
	i = 0
	for curr in sorted(currencies.keys()):
		print "["+str(i+1)+"]\t"+curr+"\t"+str(float(i)/float(len(currencies.keys())))
		curr_USD_rates = get_USD_exchange(curr)
		if curr_USD_rates != None:
			curr_USD_close = [rate['Close'] for rate in curr_USD_rates]
			if len(curr_USD_close)==261:
			#pdb.set_trace()
				f.write(curr+'\t'+'\t'.join([k for k in curr_USD_close])+"\n");
		i = i+1
	print "...done!"
	f.close();

def get_M():
	''' 
	@Desc: 	Loads exchange rates from local file "usd_xchange.txt" and parses it into matrix M
	@Returns matrix M = {<currency>: [rate1, rate2, ... , rateN], <currency2>: ..., ...}
	'''
	M = {}
	f = open('usd_xchange.txt')
	for line in f:
		split = line[:-1].split("\t")
		M[split[0]] = [float(k) for k in split[1:]]
	return M

def calc_cij(xi, xj):
	''' 
	@Desc: 	Calculates cij by formula (1) as described in the paper in section 4.2
	@Returns 
	'''
	
	# make the log lists by formula (2)
	log_xi = [math.log(xi[i]/xi[i-1]) for i in range(1,len(xi))]
	log_xj = [math.log(xj[j]/xj[j-1]) for j in range(1,len(xj))]
	# calculate cij from log_xi, log_xj by formula (1)
	xi_mean = np.mean(log_xi)
	xj_mean = np.mean(log_xj)
	scij_upper = sum((log_xi-xi_mean)*(log_xj-xj_mean))
	scij_lower = np.sqrt(sum(np.power(log_xi-xi_mean,2)))*np.sqrt(sum(np.power(log_xj-xj_mean,2)))
	return scij_upper/scij_lower


def get_A(M, currencies):	
	''' 
	@Desc: 	Calculates cross-correlation matrix A of the currencies
	@Returns matrix A = {<currency>: [a11, a12, ... , a1N], <currency2>: a21, a22, ...}
	'''
	tau = 0.5
	A = []
	
	for i in range(0,len(currencies)):
		a = []
		for j in range(0,len(currencies)):
			if i != j:	# todo: add threshold!
				cij = calc_cij(M[currencies[i]], M[currencies[j]])
				if cij>tau:
					a.append(1)
				else:
					a.append(0)
			else:
				a.append(0)
		A.append(a)
	return A
	#pdb.set_trace()
	
def export_A(A, currencies):
	''' 
	@Desc: 	exports the A matrix for Gephi
	@Returns
	'''
	f = open('A.csv', 'w+')
	f.write(";"+";".join(currencies)+"\n");
	for curr in range(0,len(currencies)):
		#pdb.set_trace()
		f.write(currencies[curr]+";"+";".join([str(a) for a in A[curr]])+"\n");
		
	f.close()
	

get_USD_rates()
M = get_M()
currencies = sorted(M.keys())
A = get_A(M, currencies)
export_A(A, currencies)


