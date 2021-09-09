import string
import re
import pycountry as pc
import pickle
import random

NUMBERS_DICT = {'1st': 'first', '2nd': 'second', '3rd': 'third', '4th': 'fourth', '5th': 'fifth', '6th': 'sixth',
               '7th': 'seventh', '8th': 'eighth', '9th': 'ninth', '10th': 'tenth'}

STANDARDS_DICT = {'3g': 'third generation', '4g': 'four genereation', '5g': 'five generation'}

with open('units_dict.pickle', 'rb') as handle:
    UNITS_DICT = pickle.load(handle)

UNITS_DICT['k'] = 'kilo'
UNITS_DICT['m'] = 'million'
UNITS_DICT['bn'] = 'billion'
UNITS_DICT['tn'] = 'trillion'
UNITS_DICT['x'] = '<rate>'
UNITS_DICT['bncapital'] = 'billion capital'
UNITS_DICT['bnequity'] = 'billion equity'
UNITS_DICT['bnextdividend'] = 'billion extra dividend'
UNITS_DICT['bnimpact'] = 'billion impact'
UNITS_DICT['ktonnes'] = 'kilo tonnes'
UNITS_DICT['eimpact'] = 'economic impact' 
UNITS_DICT['mbit'] = 'mega bit'
UNITS_DICT['mbunker'] = 'million bunker'
UNITS_DICT['mcompleted'] = 'million completed'
UNITS_DICT['mconfirms'] = 'million confirms'
UNITS_DICT['mdistrict'] = 'million district'
UNITS_DICT['mestimate'] = 'million estimate'
UNITS_DICT['mextra'] = 'million extra'
UNITS_DICT['mhigher'] = 'million higher'
UNITS_DICT['mimpact'] = 'million impact'
UNITS_DICT['mlimited'] = 'million limited'
UNITS_DICT['moil'] = 'million oil'
UNITS_DICT['morder'] = 'million order'
UNITS_DICT['more'] = 'more'
UNITS_DICT['mreversed'] = 'million reversed'
UNITS_DICT['mstrong'] = 'million strong'
UNITS_DICT['mterminals'] = 'million terminals'
UNITS_DICT['mtonnes'] = 'million tonnes'
UNITS_DICT['musd'] = 'million usd'
UNITS_DICT['mhz'] = '<units>'
UNITS_DICT['¼'] = '<frac>'
UNITS_DICT['½'] = '<frac>'

UNITS_DICT.pop('a')
UNITS_DICT.pop('am')
UNITS_DICT.pop('as')
UNITS_DICT.pop('ebit')
UNITS_DICT.pop('ex')
UNITS_DICT.pop('i')
UNITS_DICT.pop('min')
UNITS_DICT.pop('mill')
UNITS_DICT.pop('more')
UNITS_DICT.pop('no')  
UNITS_DICT.pop('of')
UNITS_DICT.pop('per')
UNITS_DICT.pop('s')

with open('financial_abbreviations.pickle', 'rb') as handle:
    FINANTIAL_ABBRV_DICT = pickle.load(handle)

FINANTIAL_ABBRV_DICT['g&a'] = 'general and administration expense'
FINANTIAL_ABBRV_DICT['ir'] = 'interest rate'
FINANTIAL_ABBRV_DICT['kpi'] = 'key performance indicator'
FINANTIAL_ABBRV_DICT['kyc'] = 'know your customer'
FINANTIAL_ABBRV_DICT['mom'] = 'month over month'
    
CURRENCY_LIST = set('AFN EUR ALL DZD USD EUR AOA XCD XCD ARS AMD AWG AUD EUR AZN BSD BHD BDT BBD BYN EUR BZD XOF BMD INR BTN BOB BOV USD BAM BWP NOK BRL USD BND BGN XOF BIF CVE KHR XAF CAD KYD XAF XAF CLP CLF CNY AUD AUD COP COU KMF CDF XAF NZD CRC XOF HRK CUP CUC ANG EUR CZK DKK DJF XCD DOP USD EGP SVC USD XAF ERN EUR SZL ETB EUR FKP DKK FJD EUR EUR EUR XPF EUR XAF GMD GEL EUR GHS GIP EUR DKK XCD EUR USD GTQ GBP GNF XOF GYD HTG USD AUD EUR HNL HKD HUF ISK INR IDR XDR IRR IQD EUR GBP ILS EUR JMD JPY GBP JOD KZT KES AUD KPW KRW KWD KGS LAK EUR LBP LSL ZAR LRD LYD CHF EUR EUR MOP MKD MGA MWK MYR MVR XOF EUR USD EUR MRU MUR EUR XUA MXN MXV USD MDL EUR MNT EUR XCD MAD MZN MMK NAD ZAR AUD NPR EUR XPF NZD NIO XOF NGN NZD AUD USD NOK OMR PKR USD  PAB USD PGK PYG PEN PHP NZD PLN EUR USD QAR EUR RON RUB RWF EUR SHP XCD XCD EUR EUR XCD WST EUR STN SAR XOF RSD SCR SLL SGD ANG XSU EUR EUR SBD SOS ZAR  SSP EUR LKR SDG SRD NOK SEK CHF CHE CHW SYP TWD TJS TZS THB USD XOF NZD TOP TTD TND TRY TMT USD AUD UGX UAH AED GBP USD USD USN UYU UYI UYW UZS VUV VES VND USD USD XPF MAD YER ZMW ZWL XBA XBB XBC XBD XTS XXX XAU XPD XPT XAG'.split())
CURRENCY_LIST = list(map(lambda x: x.lower(), list(CURRENCY_LIST)))

COUNTRY_LIST = []
for item in list(pc.countries):
    COUNTRY_LIST.append(item.name.lower())
    
CONTINENT_LIST = ['asia', 'north america', 'south america', 'africa', 'europe', 'oceania', 'antarctica',
                 'asian', 'north american', 'south american', 'african', 'european', 'oceanian', 'antarctican']

SUBREGION_LIST = ['northern africa', 'northern african',
                  'north africa', 'north african',
                  'eastern africa', 'eastern african',
                  'east africa', 'east african',
                  'middle africa', 'middle african',
                  'southern africa', 'southern african',
                  'south africa', 'south african',
                  'western africa', 'western african',
                  'west africa', 'west african',
                  
                  'northern aisa', 'northern aisan',
                  'north aisa', 'north aisan',
                  'eastern aisa', 'eastern aisan',
                  'east aisa', 'east aisan',
                  'middle aisa', 'middle aisan',
                  'southern aisa', 'southern aisan',
                  'south aisa', 'south aisan',
                  'western aisa', 'western aisan',
                  'west aisa', 'west aisan',
                  'middle east',
                  
                  'northern europe', 'northern european',
                  'north europe', 'north european',
                  'eastern europe', 'eastern european',
                  'east europe', 'east european',
                  'middle europe', 'middle european',
                  'southern europe', 'southern european',
                  'south europe', 'south european',
                  'western europe', 'western european',
                  'west europe', 'west european',
                  
                  'scandinavia', 'scandinavian',
                 ]

with open('company_names.pickle', 'rb') as handle:
    COMPANY_NAMES_LIST = pickle.load(handle)
    
COMPANY_NAMES_LIST.append('TOT')
COMPANY_NAMES_LIST.append('dtac')
COMPANY_NAMES_LIST.append('Dtac')
COMPANY_NAMES_LIST.append('CAT')
COMPANY_NAMES_LIST.append('Lehto') 



class FinancialReportsPreprocess():
    def __init__(self, random_all = True, tokens_dict=None):
        self.random_all = random_all
        self.tokens_dict = tokens_dict
        self.punctuations_pre = '!"#$()*/:;<=>?@[\\]^_`{|}~'
        self.trans_tbl_pre = ''.maketrans(self.punctuations_pre, len(self.punctuations_pre)*' ')
        self.punctuations_post = '&.,-+\''
        self.trans_tbl_post = ''.maketrans(self.punctuations_post, len(self.punctuations_post)*' ')
    
    def preprocess(self, string):
        string = self.remove_html_tags(string)
        string = self.remove_urls(string)
        string = string.translate(self.trans_tbl_pre)
        
        
        if self.tokens_dict is None:
            
            string = self.insert_company_token(string)
            

            string = string.lower()
            
            
            
            string = self.insert_financial_unabbrv_words(string)
            string = self.insert_number_unabbrv_words(string)
            string = self.insert_standard_unabbrv_words(string)
            string = self.num_unit_seperator(string)
            
            # string = self.insert_financial_unabbrv_words(string)  # no idea why it's called again
            string = self.insert_unit_unabbrv_words(string)
            
            string = self.insert_currency_token(string)

            string = self.insert_country_token(string)
            string = self.insert_continent_token(string)
            string = self.insert_subregion_token(string)


            string = self.remove_apostrophe_s(string)
            string = self.insert_Q1_token(string)
            string = self.insert_Q2_token(string)
            string = self.insert_Q3_token(string)
            string = self.insert_Q4_token(string)
            string = self.insert_date_token(string)

            string = self.insert_percentages_token(string)
            string = self.insert_positive_percentage_token(string)
            string = self.insert_negative_percentage_token(string)
            string = self.insert_percentage_token(string)
            

            string = self.insert_year_token(string)
            string = self.insert_years_token(string)
            
            
            string = self.insert_numbers_token(string)
            string = self.insert_number_token(string)
            
            string = re.sub(r'\n', ' ', string)
            string = re.sub(r'\t', ' ', string)
            
            
            string = string.translate(self.trans_tbl_post)
            return string
        else:
            
            string = self.insert_company_token(string, token=self.tokens_dict['company'])
            
            string = string.lower()
            
            
            
            string = self.insert_financial_unabbrv_words(string)
            string = self.insert_number_unabbrv_words(string)
            string = self.insert_standard_unabbrv_words(string)
            string = self.num_unit_seperator(string)

            # string = self.insert_financial_unabbrv_words(string)  # no idea why it's called again
            string = self.insert_unit_unabbrv_words(string)
            
            string = self.insert_currency_token(string, token=self.tokens_dict['currency'])

            
            string = self.insert_country_token(string, token=self.tokens_dict['country'])
            string = self.insert_continent_token(string, token=self.tokens_dict['continent'])
            string = self.insert_subregion_token(string, token=self.tokens_dict['subregion'])


            string = self.remove_apostrophe_s(string)
            string = self.insert_Q1_token(string, token=self.tokens_dict['Q1'])
            string = self.insert_Q2_token(string, token=self.tokens_dict['Q2'])
            string = self.insert_Q3_token(string, token=self.tokens_dict['Q3'])
            string = self.insert_Q4_token(string, token=self.tokens_dict['Q4'])
            string = self.insert_date_token(string, token=self.tokens_dict['date'])

            string = self.insert_percentages_token(string, token=self.tokens_dict['percentages'])
            string = self.insert_positive_percentage_token(string, token=self.tokens_dict['positive_percentage'])
            string = self.insert_negative_percentage_token(string, token=self.tokens_dict['negative_percentage'])
            string = self.insert_percentage_token(string, token=self.tokens_dict['percentage'])
            

            string = self.insert_year_token(string, token=self.tokens_dict['year'])
            string = self.insert_years_token(string, token=self.tokens_dict['years'])
            
            string = self.insert_numbers_token(string, token=self.tokens_dict['numbers'])
            string = self.insert_number_token(string, token=self.tokens_dict['number'])
            
            string = re.sub('\n', ' ', string)
            string = re.sub('\t', ' ', string)
            
            string = string.translate(self.trans_tbl_post)
            return string
        
    def remove_html_tags(self, string):
        return re.sub(r'<([a-z]+)\s[^<]+<\/\1>', '', string)
    
    def remove_urls(self, string):
        return re.sub(r'((https?|ftp|smtp):\/\/(([a-zA-Z0-9.\-_=?&\\#]+\/?)+))', '', string)
    
    def num_unit_seperator(self, string): #ex: 34,000mt
        return re.sub(r'(\b\d+([.,]\d+)?)([^-0-9\s]+\b)', r'\1 \3', string)
    
    def insert_currency_token(self, string, token='<currency>'): #ex: SEK
        currencies = []
        for currency in CURRENCY_LIST:
            if string.find(currency) != -1:
                currencies.append(currency)
        for currency in currencies:
            if self.random_all:
                string = re.sub(r'\b' + currency + r'\b', lambda x: random.choice(CURRENCY_LIST), string)
            else:
                string = re.sub(r'\b' + currency + r'\b', token, string)
        return string
    
    def insert_company_token(self, string, token='<company>'): #ex: Zolando
        companies = []
        for company in COMPANY_NAMES_LIST:
            if string.find(company) != -1:
                companies.append(company)
        for company in companies:
            if self.random_all:
                string = re.sub(r'\b' + company + r'\b', lambda x: random.choice(COMPANY_NAMES_LIST), string)
            else:
                string = re.sub(r'\b' + company + r'\b', token, string)
        return string
    
    def remove_apostrophe_s(self, string, token=''): #ex: company's
        string = re.sub(r'’s\b', token, string)
        string = re.sub(r"'s\b", token, string)
        return string

    def insert_country_token(self, string, token='<country>'): #ex:china
        countries = []
        for country in COUNTRY_LIST:
            if string.find(country) != -1:
                countries.append(country)
        for country in countries:
            if self.random_all:
                string = re.sub(r'\b' + country + r'\b', lambda x: random.choice(COUNTRY_LIST), string)
            else:
                string = re.sub(r'\b' + country + r'\b', token, string)
        return string
    
    def insert_continent_token(self, string, token='<continent>'): #ex:asia
        for continent in CONTINENT_LIST:
            if self.random_all:
                string = re.sub(r'\b' + continent + r'\b', lambda x: random.choice(CONTINENT_LIST), string)
            else:
                string = re.sub(r'\b' + continent + r'\b', token, string)
        return string
    
    def insert_subregion_token(self, string, token='<subregion>'): #ex:asia #IMPORTANT: this function should come before continent token replacements
        for subregion in SUBREGION_LIST:
            if self.random_all:
                string = re.sub(r'\b' + subregion + r'\b', lambda x: random.choice(SUBREGION_LIST), string)
            else:
                string = re.sub(r'\b' + subregion + r'\b', token, string)
        return string
    
    def insert_Q1_token(self, string, token='first quarter'): #ex:q1
        return re.sub(r'\bq1\b', token, string)
    def insert_Q2_token(self, string, token='second quarter'): #ex:q2
        return re.sub(r'\bq2\b', token, string)
    def insert_Q3_token(self, string, token='third quarter'): #ex:q3
        return re.sub(r'\bq3\b', token, string)
    def insert_Q4_token(self, string, token='fourth quarter'): #ex:q4
        return re.sub(r'\bq4\b', token, string)
    
    def insert_date_token(self, string, token='<date>'): #ex:9 february
        string = re.sub(r'\d{1,2}\s(january|february|march|april|may|june|july|august|september|october|november|december)',
                     token,
                     string)
        string = re.sub(r'\bjanuary\b|\bfebruary\b|\bmarch\b|\bapril\b|\bmay\b|\bjune\b|\bjuly\b|\baugust\b|\bseptember\b|\boctober\b|\bnovember\b|\bdecember\b',
                     token,
                     string)
        return string
    
    def insert_percentage_token(self, string, token='<percent>'): #ex:3.2%
        return re.sub(r'\b\d*\.?\d*%', token, string)
    def insert_percentages_token(self, string, token='<percents>'): #ex:3.2-4.5%
        return re.sub(r'\b(\d+\.)?\d+-(\d+\.)?\d+%', token, string)
    def insert_positive_percentage_token(self, string, token='<positive_percent>'): #ex:+3.2%
        return re.sub(r'[+]\d*\.?\d*%', token, string)
    def insert_negative_percentage_token(self, string, token='<negative_percent>'): #ex:-3.2%
        return re.sub(r'[-]\d*\.?\d*%', token, string)
    
    
    def insert_year_token(self, string, token='<year>'): #ex: 2010 
        return re.sub(r'\b19\d{2}|20\d{2}\b', token, string)
    def insert_years_token(self, string, token='<years>'): #ex: 2014-15
        return re.sub(r'\b19\d{2}|20\d{2}-\d{2}\b', token, string)
    
    def insert_number_token(self, string, token='<number>'): #ex:3.2 #IMPORTANT: This should come after all other token replacements
        return re.sub(r'[-+]?\d+([.,]\d+)?\b', token, string)
    
    def insert_numbers_token(self, string, token='<number_range> '): #ex:34.3-56.4
        return re.sub(r'\b(\d+\.)?\d+-(\d+\.)?\d+', token, string)
    
    def insert_financial_unabbrv_words(self, string):
        abbrvs = []
        for abbrv in FINANTIAL_ABBRV_DICT.keys():
            if string.find(abbrv) != -1:
                abbrvs.append(abbrv)
        for abbrv in abbrvs:
            string = re.sub(r'\b' + abbrv + r'\b', FINANTIAL_ABBRV_DICT[abbrv], string)
        return string
    
    def insert_number_unabbrv_words(self, string):
        abbrvs = []
        for abbrv in NUMBERS_DICT.keys():
            if string.find(abbrv) != -1:
                abbrvs.append(abbrv)
        for abbrv in abbrvs:
            string = re.sub(r'\b' + abbrv + r'\b', NUMBERS_DICT[abbrv], string)
        return string
    
    def insert_standard_unabbrv_words(self, string):
        abbrvs = []
        for abbrv in STANDARDS_DICT.keys():
            if string.find(abbrv) != -1:
                abbrvs.append(abbrv)
        for abbrv in abbrvs:
            string = re.sub(r'\b' + abbrv + r'\b', STANDARDS_DICT[abbrv], string)
        return string
    
    def insert_unit_unabbrv_words(self, string):
        abbrvs = []
        for abbrv in UNITS_DICT.keys():
            if string.find(abbrv) != -1:
                abbrvs.append(abbrv)
        for abbrv in abbrvs:
            string = re.sub(r'\b' + abbrv + r'\b', UNITS_DICT[abbrv], string)
        return string