import time
import random
import csv
import re
import os
import threading
import sys
import signal

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager

class TripAdvisorScraper:
    def __init__(self, url, max_reviews=52658):
        self.url = url
        self.max_reviews = max_reviews
        self.total_reviews_scraped = 0
        self.driver = self.setup_driver()
        self.reviews = []
        self.actions = ActionChains(self.driver)
        
        # Control flags
        self.is_paused = False
        self.is_stopped = False

    def setup_driver(self):
        """Setup and configure Selenium WebDriver"""
        options = Options()
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36"
        ]
        options.add_argument(f"user-agent={random.choice(user_agents)}")

        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=options)

    def toggle_pause(self):
        """Toggle pause state"""
        self.is_paused = not self.is_paused
        print(f"\n{'⏸️ Pausing' if self.is_paused else '▶️ Resuming'} scraping process...")

    def stop(self):
        """Stop the scraping process"""
        print("\n⏹️ Stopping scraping process...")
        self.is_stopped = True
        self.driver.quit()

    def check_pause(self):
        """Check if scraping is paused"""
        while self.is_paused:
            time.sleep(0.5)
            if self.is_stopped:
                break

    def scroll_page(self):
        """Scroll the page to load more reviews"""
        # Check for pause state
        self.check_pause()

        # Scroll to the bottom of the page
        try:
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            while True:
                # Check if stopped or paused
                if self.is_stopped:
                    break
                self.check_pause()

                # Scroll down to bottom
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                
                # Wait to load page
                time.sleep(2)
                
                # Calculate new scroll height and compare with last scroll height
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                
                # Break the loop if no more scrolling is possible
                if new_height == last_height:
                    break
                
                last_height = new_height
        except Exception as e:
            print(f"Error during scrolling: {e}")

    def extract_rating(self, rating_svg):
        """Extract numeric rating from SVG title"""
        if rating_svg is None:
            return "N/A"
        
        title_elem = rating_svg.find("title")
        if title_elem:
            match = re.search(r'^(\d+(?:\.\d+)?)', title_elem.get_text(strip=True))
            return match.group(1) if match else "N/A"
        return "N/A"

    def preprocess_route(self, route):
        """Preprocess route to extract clean origin/destination"""
        return route.split(" - ")[0].strip()

    def scrape_current_page(self):
        """Scrape reviews from current page"""
        # Check for pause
        self.check_pause()

        # Parse the page source
        soup = BeautifulSoup(self.driver.page_source, "html.parser")

        # Define class names
        CLASS_REVIEW_BLOCK = "lwGaE A"
        CLASS_REVIEW_TEXT = "JguWG"
        CLASS_FLIGHT_TYPE = "thpSa"
        CLASS_CATEGORY_CONTAINER = "msVPq"
        CLASS_CATEGORY_RATING_CONTAINER = "f Q1 u"
        CLASS_RATING_SVG = "UctUV d H0"
        CLASS_CATEGORY_TITLE = "biGQs _P pZUbB osNWb"

        # Find all review blocks
        page_reviews = []
        for review_block in soup.find_all("div", class_=CLASS_REVIEW_BLOCK):
            review_data = {}

            # Extract review text
            review_text_elem = review_block.find("span", class_=CLASS_REVIEW_TEXT)
            review_data["Review Text"] = review_text_elem.get_text(strip=True).replace('"', "'") if review_text_elem else "N/A"

            # Extract origin and destination
            origin_destination_elems = review_block.find_all("span", class_=CLASS_FLIGHT_TYPE)
            if len(origin_destination_elems) >= 2:
                review_data["Origin"] = self.preprocess_route(origin_destination_elems[0].get_text(strip=True))
                dest_text = origin_destination_elems[1].get_text(strip=True)

                # Split destination to separate flight type
                dest_parts = dest_text.split()
                if len(dest_parts) > 1:
                    review_data["Destination"] = dest_parts[0]
                    review_data["Flight Type"] = " ".join(dest_parts[1:])
                else:
                    review_data["Destination"] = origin_destination_elems[0].get_text(strip=True).split(" - ")[1]
                    review_data["Flight Type"] = dest_text
            else:
                review_data["Origin"] = "N/A"
                review_data["Destination"] = "N/A"
                review_data["Flight Type"] = "N/A"

            # Extract date of travel
            date_of_travel_elem = review_block.find("b")  
            review_data["Date of Travel"] = date_of_travel_elem.get_text(strip=True) if date_of_travel_elem else "N/A"

            # Extract overall rating
            rating_svg = review_block.find("svg", class_=CLASS_RATING_SVG)
            review_data["Overall Rating"] = self.extract_rating(rating_svg)

            # Predefined categories to ensure all are extracted
            categories = [
                "Legroom", "Seat Comfort", "In-flight Entertainment", 
                "Customer Service", "Value for Money", "Cleanliness", 
                "Check-in and Boarding", "Food and Beverage"
            ]

            # Initialize all categories to N/A
            for category in categories:
                review_data[category] = "N/A"
            
            # Extract category ratings
            category_containers = review_block.find_all("div", class_=CLASS_CATEGORY_CONTAINER)
            
            for container in category_containers:
                rating_container = container.find("div", class_=CLASS_CATEGORY_RATING_CONTAINER)
                if rating_container:
                    rating_svg = rating_container.find("svg", class_=CLASS_RATING_SVG)
                    category_title = rating_container.find("div", class_=CLASS_CATEGORY_TITLE)

                    if rating_svg and category_title:
                        category_name = category_title.get_text(strip=True)
                        category_rating = self.extract_rating(rating_svg)
                        
                        # Map similar categories
                        if category_name == "Seat comfort":
                            category_name = "Seat Comfort"
                        elif category_name == "Customer service":
                            category_name = "Customer Service"
                        elif category_name == "Check-in and boarding":
                            category_name = "Check-in and Boarding"
                        elif category_name == "Value for money":
                            category_name = "Value for Money"
                        
                        review_data[category_name] = category_rating

            page_reviews.append(review_data)

        return page_reviews

    def click_next_page(self):
        """Click the Next Page button with multiple strategies"""
        try:
            # Check for pause
            self.check_pause()

            # Find the next page button using multiple strategies
            next_buttons = self.driver.find_elements(By.XPATH, "//button[@aria-label='Next page']")
            
            if not next_buttons:
                # Alternative selector if the first doesn't work
                next_buttons = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'IGLCo')]//button")
            
            if next_buttons:
                # Try multiple clicking strategies
                button = next_buttons[0]
                
                try:
                    # Strategy 1: Direct Selenium click
                    button.click()
                except Exception:
                    try:
                        # Strategy 2: JavaScript click
                        self.driver.execute_script("arguments[0].click();", button)
                    except Exception:
                        # Strategy 3: Action Chains click
                        self.actions.move_to_element(button).click().perform()
                
                # Wait after clicking
                time.sleep(random.uniform(3, 5))
                return True
            
            print("No next page button found")
            return False
        
        except Exception as e:
            print(f"Error clicking next page: {e}")
            return False

    def save_to_csv(self, filename="tripadvisor_reviews.csv"):
        """Save reviews to CSV, creating file if it doesn't exist and appending reviews"""
        headers = [
            "Review Text", "Overall Rating", "Origin", "Destination", "Flight Type", "Date of Travel",
            "Legroom", "Seat Comfort", "In-flight Entertainment", "Customer Service",
            "Value for Money", "Cleanliness", "Check-in and Boarding", "Food and Beverage"
        ]

        # Determine if file needs headers
        file_exists = os.path.exists(filename)

        # Open file in append mode (will create if not exists)
        with open(filename, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            
            # Write headers only if file is new
            if not file_exists:
                writer.writeheader()
            
            # Write current page's reviews
            for review in self.reviews:
                writer.writerow({key: review.get(key, "N/A") for key in headers})
        
        print(f"✅ Reviews {'appended to' if file_exists else 'saved in'} {filename}")

    def scrape_reviews(self):
        """Main scraping method with pause/resume/stop capabilities"""
        try:
            # Navigate to the page
            self.driver.get(self.url)
            
            # Print instructions
            print("\n🔍 Scraping started!")
            print("Controls:")
            print("  Ctrl+C: Stop scraping")
            print("  Enter key: Pause/Resume")
            
            # Initial wait and scroll
            time.sleep(5)
            self.scroll_page()

            # Continue scraping until max reviews or no more pages
            while self.total_reviews_scraped < self.max_reviews and not self.is_stopped:
                # Check for pause
                self.check_pause()

                # Scrape current page
                current_page_reviews = self.scrape_current_page()
                self.reviews.extend(current_page_reviews)
                self.total_reviews_scraped += len(current_page_reviews)
                
                # Save current page's reviews to CSV immediately
                self.save_to_csv()
                
                print(f"Scraped {self.total_reviews_scraped} reviews")

                # Clear reviews to prevent memory issues
                self.reviews.clear()

                # Try to click next page
                if not self.click_next_page():
                    print("No more pages available or unable to click next page.")
                    break

                # Additional scroll and wait
                self.scroll_page()
                time.sleep(random.uniform(2, 4))

        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if not self.is_stopped:
                self.driver.quit()

def input_thread(scraper):
    """Thread to handle user input"""
    try:
        while not scraper.is_stopped:
            input()  # Wait for Enter key
            scraper.toggle_pause()
    except:
        pass

def main():
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    tripadvisor_url = "https://www.tripadvisor.com/Airline_Review-d8729039-Reviews-British-Airways"
    
    # Create scraper instance
    scraper = TripAdvisorScraper(tripadvisor_url)
    
    # Create input thread
    input_monitoring = threading.Thread(target=input_thread, args=(scraper,), daemon=True)
    input_monitoring.start()
    
    # Run scraper in main thread
    scraper.scrape_reviews()

if __name__ == "__main__":
    main()