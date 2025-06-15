import requests
from bs4 import BeautifulSoup
import csv
import os
from datetime import datetime

url = "https://id.tradingview.com/markets/world-stocks/worlds-largest-companies/"

response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")

    price_elements = soup.find_all('td', class_='right-RLhfr_y4')
    company_code = soup.find_all('a', class_='apply-common-tooltip tickerNameBox-GrtoTeat tickerName-GrtoTeat')
    company_name = soup.find_all('sup', class_="apply-common-tooltip tickerDescription-GrtoTeat")
    
    #menyimpan file CSV
    output_path = 'C:/Users/aa807/Desktop/big data/HargaSaham.csv'  # Menyimpan dengan nama file tetap HargaSaham.csv

    file_exists = os.path.exists(output_path)

    with open(output_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Nama Perusahaan', 'Kode Perusahaan', 'Harga Saham', 'Market Cap'])
        temp = True
        for i in range(1, 11):  #mengambil data untuk 10 perusahaan pertama
            if len(company_code) > i and len(price_elements) > i and len(company_name) > i:
                company_code_text = company_code[i-1].get_text(strip=True)  
                company_name_text = company_name[i-1].get_text(strip=True)  
                
                # tambah timestamp 
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                if temp is True:
                    price = price_elements[i].get_text(strip=True)  
                    Market_cap = price_elements[i-1].get_text(strip=True)
                    temp = False
                else:
                    price = price_elements[((i-1)*8)+1].get_text(strip=True)  
                    Market_cap = price_elements[(i-1)*8].get_text(strip=True)
                
                # tulis CSV 
                writer.writerow([timestamp, company_name_text, company_code_text, price, Market_cap])
                print(f"Data perusahaan {i} ditulis ke CSV")
            else:
                print(f"Data tidak ditemukan untuk perusahaan nomor {i+1}.")
else:
    print(f"Gagal mengakses halaman. Status code: {response.status_code}")
