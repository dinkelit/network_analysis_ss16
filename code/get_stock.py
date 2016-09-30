import urllib, json, pdb, math, numpy as np, networkx as nx, matplotlib.pyplot as plt, operator, collections, time, itertools

# All this code was written by Viktor Dinkel

def download_Stock_rates(company):
	''' 
	@Desc: 	Loads the stock rates of a company from yahoo.finance.historicaldata
	@Param: company (String) ex: "CFO"
	@Returns sorted (ascending) list of dictionaries {'Volume': <val>, 'Open': <val>, 'Close': <val>, ...}
	'''
	qString = 'select * from yahoo.finance.historicaldata where symbol = "'+company+'" and startDate = "2015-01-01" and endDate = "2016-01-01"';
	qString_encoded = urllib.quote_plus(qString)

	url = "https://query.yahooapis.com/v1/public/yql?q="+qString_encoded+"&format=json&diagnostics=true&env=store%3A%2F%2Fdatatables.org%2Falltableswithkeys&callback="
	response = urllib.urlopen(url)
	data = json.loads(response.read())
	if data['query']['results'] == None:
		return None
	else:
		ret = list(reversed(data['query']['results']['quote']))
		return ret

def get_companies():
	''' 
	@Desc: 	Loads the company name & abbrevation from local file 
	@Returns dictionary = {abbrevation: {name: <name>}, ...}
	'''
	f = open("companylist.csv")
	companies = {}
	got_header = False
	for line in f:
		if not got_header:
			header = line.replace("\"","").replace("\'","").replace('\n','').replace('\r','').split(',')[:-1]
			got_header = True
		else:
			split = line[1:].replace('\n','').replace('\r','').split('\",\"')[:-1]
			companies[split[0]] = {"name": split[1]}
	return companies

def get_Stock_rates():
	''' 
	@Desc: 	Writes file with rate matrix of companies.
	@Returns void, local file "stock_data.csv" with content: <currency>\t<rate1>\t<rate2>...\t<rateN>\n
	'''

	f = open('stock_data.txt', 'a+')
	companies = get_companies()
	i = 0
	close_counts = {} 

	# continue where you last stopped
	g = open('continue_here.txt', 'r+')
	continue_here = ""
	for line in g:
		continue_here = line.replace("\n","")
	if len(continue_here) == 0:
		continue_here = sorted(companies.keys())[0]

	start_downloading = False
	for company in sorted(companies.keys()):
		try:
			if start_downloading:
				print "["+str(i+1)+"]\t"+company+"\t"+str(float(i)/float(len(companies.keys())))
				Stock_rates = download_Stock_rates(company)
				if Stock_rates != None:
					Stock_close = [rate['Close'] for rate in Stock_rates]
					if len(Stock_close)==252:
						f.write(company+'\t'+'\t'.join([k for k in Stock_close])+"\n");
				i = i+1
				try:
					close_counts[str(len(Stock_close))] = close_counts[len(Stock_close)]+1
				except:
					close_counts[str(len(Stock_close))] = 1
		
			if company == continue_here:
				start_downloading = True
		except:
			print "--- Error, try again"
			g.write(company)
			g.close()
			f.close()
			
			
	print "...done!"
	pdb.set_trace()
	f.close()
	g.close()

def get_M():
	''' 
	@Desc: 	Loads rates from local file "stock_data.txt" and parses it into matrix M
	@Returns matrix M = {<company>: [rate1, rate2, ... , rateN], <company2>: ..., ...}
	'''
	M = {}
	f = open('stock_data.txt')
	for line in f:
		split = line[:-1].split("\t")
		#pdb.set_trace()
		M[split[0]] = [math.log(float(split[k])/float(split[k-1])) for k in range(2,len(split))] 
	return M

def calc_cij(xi, xj):
	''' 
	@Desc: 	Calculates cij by formula (1) as described in the paper in section 4.2
	@Returns 
	'''
	
	# make the log lists by formula (2)
	log_xi = []
	for i in range(1,len(xi)):
		try:
			log_xi.append(math.log(xi[i]/xi[i-1]))
		except:
			log_xi.append(0.01)

	log_xj = []
	for j in range(1,len(xj)):
		try:
			log_xj.append(math.log(xi[j]/xj[j-1]))
		except:
			log_xj.append(0.01)
	# calculate cij from log_xi, log_xj by formula (1)
	xi_mean = np.mean(log_xi)
	xj_mean = np.mean(log_xj)
	scij_upper = sum((log_xi-xi_mean)*(log_xj-xj_mean))
	scij_lower = np.sqrt(sum(np.power(log_xi-xi_mean,2)))*np.sqrt(sum(np.power(log_xj-xj_mean,2)))
	return scij_upper/scij_lower


def get_M_log(M, companies):	
	''' 
	@Desc: 	Calculates cross-correlation matrix A of the companies
	@Returns matrix A = {<company>: [a11, a12, ... , a1N], <company2>: a21, a22, ...}
	'''
	A = []
	
	for i in range(0,len(companies)):
		print i
		a = []
		for j in range(0,len(companies)):
			if i != j:
				cij = calc_cij(M[companies[i]], M[companies[j]])
				a.append(cij)
		A.append(a)
	return A

	
def export_M_log(M_log, companies):
	''' 
	@Desc: 	exports the A matrix for Gephi
	@Returns void
	'''
	z = open('M_log.csv', 'w+')	
	[z.write(companies[company]+";"+";".join([str(a) for a in M_log[company]])+"\n") for company in range(0,len(companies))]
	z.close() 

	

def create_M_log(new_download):
	''' 
	@Desc: 	big function to handle download of stock data to export of the adjacency matrix
	@Returns void
	'''
	if (new_download):
		get_Stock_rates()
	M = get_M()
	companies = sorted(M.keys())
	M_log = get_M_log(M, companies)
	pdb.set_trace()
	export_M_log(M_log, companies)

def initialize_G(tau):
	''' 
	@Desc: 	Loads correlation-matrix and adjacency matrix (tau-specific) into G 
	@Returns Graph G
	'''
	M_log = open("M_log.csv")
	A_tau = open("A_"+str(tau)+".csv","w+")
	#A = {}
	header = []
	ii = 0
	print "creating A"
	for m in M_log:
		if m[0] == ';':
			header = m.replace('\n','').split(';')[1:]
		else:
			aa = [1 if float(k)>=tau else 0 for k in m.replace('\n','').split(';')[1:]]
			#pdb.set_trace()
			for adj in range(0,len(aa)):
				if aa[adj] == 1:
					A_tau.write(str(header[ii])+"\t"+str(header[adj])+"\n")
			ii += 1

	print "initializing G..."
	G = nx.Graph()
	f = open("A_"+str(tau)+".csv")
	nx.read_edgelist(f, encoding='utf-8', delimiter='\t', create_using=G)
	print "done, loading A"
	#A = (nx.adjacency_matrix(G)).todense()
	print "done"
	f.close()

	n = len(list(G.nodes()))

	#degree_sequence=sorted([d for n,d in G.degree()], reverse=True) # degree sequence
	#degrees_with_keys=sorted([(d,n) for n,d in G.degree()], reverse=True) # degree sequence
	
	return G


def plot_G(G, highlight = None, partition = None):
	''' 
	@Desc: 	deprecated function to plot G with highlighted nodes
	@Returns void
	'''
	dd = [d for d in nx.degree(G)]
	

	nodes = [k for k in G.nodes()]
	highlightcolors = ['green' if node in highlight else 'red' for node in nodes]
	degrees = [G.degree(node) for node in nodes]

	#nx.draw(G,pos=nx.spring_layout(G), node_size=[v * 100 for v in dd.values()], with_labels=True) # use spring layout
	nx.draw(G,nodelist = nodes, pos=nx.spring_layout(G, k = 2), node_size=[(600 + 50*v) for v in degrees], with_labels=True, node_color = highlightcolors, alpha=0.8) # use spring layout
	plt.draw()
	plt.show()

def plot_degree_distribution(G):
	''' 
	@Desc: 	plots the degree distribution and the loglog of it
	@Returns void
	'''
	degree_sequence=sorted([d for n,d in G.degree()], reverse=True) # degree sequence
	degreeCount=collections.Counter(degree_sequence)
	deg, cnt = zip(*degreeCount.items())
	plt.bar(deg, cnt, width=0.80, color='b')
	plt.show()

	A = nx.adjacency_matrix(G).todense()
	k = [d.sum() for d in A]
	p_k = [pow(d, -0.8) for d in k]
	plt.loglog(k, p_k)
	plt.show()

	#pdb.set_trace()
	
def initialize_centrality(companies, G):
	''' 
	@Desc: 	initializes the nodes with centrality values
	@Returns G with centrality values
	'''

	betweenness = nx.betweenness_centrality(G)
	closeness = nx.closeness_centrality(G)
	degree = nx.degree_centrality(G)
	for company in companies:
		G.node[company]['centrality'] = {'degree': degree[company], 'betweenness': betweenness[company], 'closeness': closeness[company]}
	return G

def centrality_function(companies, G):
	''' 
	@Desc: 	calculates the weighed Cavg for each node
	@Returns centrality dictionary with companies as keys
	'''
	max_key = ''
	max_val = 0.0
	b1 = 0.3
	b2 = 0.2
	b3 = 0.5
	centrality = {}
	for company in companies:
		val = b1*float(G.node[company]['centralities']['degree'])+b2*float(G.node[company]['centralities']['betweenness'])+b3*float(G.node[company]['centralities']['closeness'])
		centrality[company] = val
		G.node[company]['centralities']['score'] = val
		if val > max_val:
			max_val = val
			max_key = company
	
	return centrality

def get_portfolio(centralities):
	''' 
	@Desc: 	sorts the nodes by centralities, highest first
	@Returns sorted nodes by highest centralities
	'''
	centrality_sequence=sorted(centralities.items(), key=operator.itemgetter(1), reverse=True)
	return centrality_sequence

def load_meta(G):
	''' 
	@Desc: 	loads meta-data (.csv coming from Gephi) into G, like label & modularity class
	@Returns G having nodes full of information
	'''
	f = open("stock_G_0_6 [Nodes].csv")
	header = ""
	for line in f:
		#pdb.set_trace()	
		split = line.replace("\n","").replace("\r","").replace("&#39;","\'").split(";")
		if len(header)==0:
			header = split
		else:
			G.node[split[0]] = {
					"label": split[1], 
					"modularity_class": split[3], 
					"sector": split[10],
					"industry": split[11],
					"close_data": None,
					"volume_data": None,
					"centralities": None
					}
			G.node[split[0]]["loaded"] = True

	# set loaded = False for nodes which have no meta data loaded
	for node in G.nodes():
		try:
			if G.node[node]["loaded"]:
				pass
			else:
				G.node[node]["loaded"] = False
		except:
			G.node[node]["loaded"] = False

	# load closing prices
	f.close()
	f = open("data_2015/close_data.txt")
	for line in f:	
		split = line.replace("\n","").split("\t")
		try:
			G.node[split[0]]['close_data'] = [float(k) for k in split[1:]]
		except:
			pass
	f.close()

	# load stock volumes
	f = open("data_2015/volume_data.txt")
	for line in f:	
		split = line.replace("\n","").split("\t")
		try:
			G.node[split[0]]['volume_data'] = [int(k) for k in split[1:]]
		except:
			pass
	f.close()

	# load centralities
	f = open("stock_G_0_6_centralities.csv")
	read = False
	for line in f:
		if not read:
			read = True
			header = line
		else:
			split = line.replace("\n","").replace("\r","").split(";")
			try:
				G.node[split[0]]['centralities'] = {
					'degree': split[1],
					'closeness': float(split[2])*200.0,
					'betweenness': float(split[3])*0.01
					}
			except:
				pass
	f.close()
	
	return G

def sector_analysis(G):
	''' 
	@Desc: 	traverses the meta-data of G and gathers statistics regarding sectors
	@Returns void
	'''
	largest_modules = [0,5,7,9]
	module_sector_data = {"0": {}, "5": {}, "7": {}, "9": {}}
	module_sector_percent = {"0": {}, "5": {}, "7": {}, "9": {}}
	global_sector_data = {}
	global_sector_percent = {}
	total = 0
	# count global sector occurences
	for node in G.nodes():
		node_sector = G.node[node]['sector']
		node_module = G.node[node]['modularity_class']
		total += 1
		try:
			global_sector_data[node_sector] += 1
		except:
			global_sector_data[node_sector] = 1

		if (G.node[node]["loaded"]):
			if int(node_module) in largest_modules:
				try:
					module_sector_data[node_module][node_sector] += 1
				except:
					module_sector_data[node_module][node_sector] = 1
			
	# calculate global sector percentage
	for sector in global_sector_data.keys():
	 	global_sector_percent[sector] = float(global_sector_data[sector])/float(total) 

	for module in module_sector_data.keys():
		total_module = sum([module_sector_data[module][sector] for sector in module_sector_data[module].keys()])

		for sector in module_sector_data[module]:
			#pdb.set_trace()
	 		module_sector_percent[module][sector] = float(module_sector_data[module][sector])/float(total_module)
	# sum([global_sector_percent[sector] for sector in global_sector_percent.keys()])

	f = open("sector_analysis.csv","w+")
	f.write("module;sector;percent\n")
	for module in module_sector_percent.keys():
		for sector in module_sector_percent[module]:
			f.write(module+";"+sector+";"+str(module_sector_percent[module][sector])+"\n")
			#pdb.set_trace()
	f.close()

def calculate_capital_return(G, modules = None):
	invalid = 0
	for node in [k for k in G.nodes()]:
		capital_return = []
		for i in range(0,len(G.node[node]['close_data'])):

			if G.node[node]['volume_data'] != None:
				if len(G.node[node]['volume_data'])!=252:
					G.node[node]['valid'] = False
				else:
					G.node[node]['valid'] = True

				close_t = float(G.node[node]['close_data'][i])
				vol_t = float(G.node[node]['volume_data'][i])
				close_ts = float(G.node[node]['close_data'][0])
				vol_ts = float(G.node[node]['volume_data'][0])
				try:
					cap_return = (vol_t*close_t-vol_ts*close_ts)/(vol_ts*close_ts) #(5)
				except:
					cap_return = 0
				capital_return.append(cap_return)

			else:
				G.node[node]['valid'] = False

		G.node[node]['capital_return'] = capital_return
		
	valid_nodes = []
	total_capital_return = []
	module_capital_return = {'0': [],'5': [],'7': [],'9': []}
	for node in [k for k in G.nodes()]:
		if G.node[node]['valid']:
			valid_nodes.append(node)	
			if len(total_capital_return) == 0:
				total_capital_return = G.node[node]['capital_return']
			else:
				total_capital_return = map(sum, zip(total_capital_return,G.node[node]['capital_return']))

			if modules and node in modules['0']+modules['5']+modules['7']+modules['9']:
				node_module = G.node[node]['modularity_class']
				if len(module_capital_return[node_module]) == 0:
					module_capital_return[node_module] = G.node[node]['capital_return']
				else:
					module_capital_return[node_module] = map(sum, zip(module_capital_return[node_module],G.node[node]['capital_return']))
		
	#normalize capital return of all stocks 
	normalized_total_capital_return = [k/len(valid_nodes) for k in total_capital_return]

	#normalize capital return of modules
	normalized_modules_capital_return = {'0': [],'5': [],'7': [],'9': []}
	for module in modules.keys():
		normalized_modules_capital_return[module] = [k/len(modules[module]) for k in module_capital_return[module]]

	plt.plot(normalized_total_capital_return, color = 'black', linewidth=2.0)
	plt.plot(normalized_modules_capital_return['9'], color= '#7D26CD')	#violet
	plt.plot(normalized_modules_capital_return['0'], color= '#4A7023')	#green
	plt.plot(normalized_modules_capital_return['7'], color= '#FF00FF')	#pink
	plt.plot(normalized_modules_capital_return['5'], color= '#B3432B')	#brown
	plt.show()
	return G, normalized_total_capital_return
	
def portfolio_evaluation(G,total_capital_return,centrality_sequence):
	''' 
	@Desc: 	traverses the meta-data of G and gathers statistics regarding capital return
	@Returns void
	'''
	portfolio_capital_return = []
	top = 20
	for k in range(0,top):
		node = centrality_sequence[k][0]
		if len(portfolio_capital_return)==0:
			portfolio_capital_return = G.node[node]['capital_return']
		else:
			portfolio_capital_return = map(sum, zip(portfolio_capital_return,G.node[node]['capital_return']))

	normalized_portfolio_capital_return = [k/top for k in portfolio_capital_return]

	plt.plot(total_capital_return, color = 'black', linewidth=2.0)
	plt.plot(normalized_portfolio_capital_return, color = 'r')
	plt.show()

	for k in [str(k[0])+"\t"+str(k[1]) for k in centrality_sequence[:20]]:
		print k
	#pdb.set_trace()

# MAIN FUNCTION -------------------------------------------------

#create_M_log(new_download = False)
G = initialize_G(0.6)
G = load_meta(G)

#sector_analysis(G)
#G = initialize_centrality([k for k in G.nodes()],G)
centralities = centrality_function([k for k in G.nodes()],G)
centrality_sequence = get_portfolio(centralities)

largest_modules = ['0','5','7','9']
modules = {'0':[], '5':[], '7':[], '9':[]}
[modules[G.node[node]['modularity_class']].append(node) if G.node[node]['modularity_class'] in largest_modules else '' for node in G.nodes()]

G, total_capital_return = calculate_capital_return(G, modules)
portfolio_evaluation(G,total_capital_return, centrality_sequence)

#plot_G(G, highlight = [k[0] for k in centrality_sequence][:10])
#plot_degree_distribution(G)

#-----------------------------------------------------------------	



