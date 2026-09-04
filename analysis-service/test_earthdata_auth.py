"""Quick test of Earthdata authentication."""
import sys
import os
sys.path.insert(0, '.')

# Load .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

from app.services.earthdata_auth import (
    get_earthdata_credentials,
    get_earthdata_auth_header,
    get_earthdata_userpwd,
    is_earthdata_url,
    test_earthdata_authentication,
    test_rasterio_earthdata_read,
)

print("=== Credential Loading ===")
user, passwd = get_earthdata_credentials()
print(f"user present: {user is not None and len(user) > 0}")
print(f"passwd present: {passwd is not None and len(passwd) > 0}")
if user:
    print(f"user length: {len(user)}")
if passwd:
    print(f"passwd length: {len(passwd)}")

print("\n=== URL Detection ===")
test_url = "https://lpdaac.earthdata.nasa.gov/HLS.S30.T43RGM.2024059T052749.v2.0.B04.tif"
print(f"is_earthdata_url(HLS): {is_earthdata_url(test_url)}")
print(f"is_earthdata_url(PlanetaryComputer): {is_earthdata_url('https://planetarycomputer.microsoft.com/api/stac/v1')}")

print("\n=== Auth Header ===")
header = get_earthdata_auth_header()
print(f"header present: {header is not None}")
if header:
    print(f"header keys: {list(header.keys())}")
    # Don't print the actual header value for security

print("\n=== Userpwd ===")
userpwd = get_earthdata_userpwd()
print(f"userpwd present: {userpwd is not None}")
if userpwd:
    print(f"userpwd length: {len(userpwd)}")
    print(f"userpwd format OK: {':' in userpwd}")

print("\n=== HTTP Authentication Test ===")
auth_result = test_earthdata_authentication()
print(f"authenticated: {auth_result['authenticated']}")
print(f"http_status: {auth_result['http_status']}")
print(f"error: {auth_result['error']}")
print(f"url (truncated): {auth_result['url'][:80]}...")

print("\n=== Rasterio Read Test ===")
read_result = test_rasterio_earthdata_read()
print(f"readable: {read_result['readable']}")
print(f"bands: {read_result['bands']}")
print(f"shape: {read_result['shape']}")
print(f"crs: {read_result['crs']}")
print(f"resolution: {read_result['resolution']}")
print(f"dtype: {read_result['dtype']}")
print(f"nodata: {read_result['nodata']}")
print(f"error: {read_result['error']}")
print(f"url (truncated): {read_result['url'][:80]}...")
