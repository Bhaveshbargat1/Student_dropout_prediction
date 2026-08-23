from flask import Flask
from flask import render_template
from flask import request

from recommender import RecommendationSystem

app = Flask(__name__)

model = RecommendationSystem(
    "dataset/transactions.csv"
)


@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []

    cart = ""

    if request.method == "POST":

        cart = request.form["cart"]

        products = [
            item.strip()
            for item in cart.split(",")
        ]

        recommendations = model.recommend(products)

    product_images = {
        "Bread": "bread.jpg",
        "Butter": "butter.jpg",
        "Milk": "milk.jpg",
        "Eggs": "eggs.jpg",
        "Laptop": "laptop.jpg",
        "Mouse": "mouse.jpg",
        "Keyboard": "keyboard.jpg",
        "Laptop Bag": "laptop_bag.jpg",
        "USB Drive": "usb_drive.jpg"
    }
    
    return render_template(
        "index.html",
        recommendations=recommendations,
        cart=cart,
        product_images=product_images
    )


if __name__ == "__main__":

    app.run(debug=True)
