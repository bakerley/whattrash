import tensorflow as tf
import numpy as np

model = None
output_class = ["Batteries", "Clothes", "E-waste", "Glass", "Light Blubs", "Metal", "Organic", "Paper", "Plastic"]
data = {
"Batteries":
	["Recycling single-use batteries is an easy way to make the world a little greener. Every battery contains some reusable material, whether rechargeable or single-use. You can either take your batteries into a local store or recycling facility, or you can mail them to a facility through a mail-in program. When you recycle your batteries, you help reduce soil contamination and water pollution, so keep recycling and making the world a better place!",
	"https://www.energizer.com/responsibility/battery-recycling/where-to-recycle-batteries", "Recycle program at many stores"],
"Clothes":
	["By donating, reselling, or disposing of textiles with reputable private recycling companies, clothes recycling is definitely better than simply throwing away unwanted items. Still, the main thing we all can do is consume less. Think before you buy: Do you really need that shirt, even if it’s on sale? Or what about those pants you’re eyeing online that look just like the three you have at home? A quick inventory of your closet will help curb those impulse buys.",
	"https://greenactioncentre.ca/reduce-your-waste/how-to-recycle-your-clothes/", "Charity Donations, Consignment Stores"],
"E-waste":
	["The mantra of \"Reduce, Reuse, Recycle\" applies here. Reduce your generation of e-waste through smart procurement and good maintenance. Reuse still functioning electronic equipment by donating or selling it to someone who can still use it. Recycle those products that cannot be repaired. Computer monitors, televisions and other electronic equipment should NOT be disposed of with regular garbage, as this is illegal in California. To find an organization that will manage your electronics for recycling, search the directory.",
	"https://www.epa.gov/international-cooperation/cleaning-electronic-waste-e-waste","Repair, Drop at recycling programs"],
"Glass":
	["Recycling may seem like a waste of time to certain people, but it certainly has some benefits if you bother to take the time. Glass containers under 24 ounces are worth 5 cents each, and containers that are 24 ounces or more can get you 10 cents. So take a look at your bottles and see if they have a CRV, or California Redemption Value, because most of them will. Once you get a decent amount saved up, toss them in your car and hustle on over to a glass recycling center near you.",
	"https://www.gpi.org/glass-recycling-facts", "Special garbage"],
"Light Blubs":
	["Compact fluorescent light (CFL) bulbs contain a very small amount of mercury, so do not put used CFLs in the garbage. While the amount of mercury in a single bulb is extremely small, it will harm the environment if large numbers of bulbs are disposed of in landfills. Select PSE offices, participating retail locations and county household-hazardous-waste facilities recycle CFL, incandescent, and LED bulbs for free. Review the list below to find a location near you. Linear fluorescent tubes are not accepted for recycling at these collection stations.",
	"https://www.epa.gov/cfl/recycling-and-disposal-cfls-and-other-bulbs-contain-mercury", "Retail locations"],
"Metal":
	["Scrap metal is one of the most valuable materials you can recycle, and it encompasses so many consumer products. From appliances to batteries to cans to clothes hangers, metal is everywhere in our homes. Recycling metal is important to not only keep this limited supply material out of landfills, but also because it can make you money.",
	"https://earth911.com/recycling-guide/how-to-recycle-metal/", "Retail location, Municipality"],
"Organic":
	["Organic wastes contain materials which originated from living organisms. There are many types of organic wastes and they can be found in municipal solid waste , industrial solid waste , agricultural waste, and wastewaters. Organic wastes are often disposed of with other wastes in landfills or incinerators, but since they are biodegradable , some organic wastes are suitable for composting and land application.<br><br>Organic materials found in municipal solid waste include food, paper, wood, sewage sludge , and yard waste.",
	"https://millerrecycling.com/organic-waste-and-how-to-handle-it/", "Municipality, Compost"],
"Paper":
	["Paper Recycling is relatively straightforward and has been around for decades. After collection, paper and cardboard is sorted then baled for shipment to a mill. At a paper mill, recycled paper products are introduced into a pulper as one of three feedstocks for making new paper: mill broke, pre-consumer waste, and post-consumer waste. Mill broke is paper recycled from the production of paper at the mill. ",
	"https://worldpapermill.com/ultimate-guide-paper-recycling-process/", "Municipality"],
"Plastic":
	["Plastic recycling refers to the process of recovering waste or scrap plastic and reprocessing the materials into functional and useful products. This activity is known as the plastic recycling process. The goal of recycling plastic is to reduce high rates of plastic pollution while putting less pressure on virgin materials to produce brand new plastic products. This approach helps to conserve resources and diverts plastics from landfills or unintended destinations such as oceans.",
	"https://plasticsrecycling.org", "Municipality, Reuse"]
}


def load_artifacts():
    global model
    model = tf.keras.models.load_model("classifyWaste.h5")

def classify_waste(image_path):
	global model, output_class
	test_image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
	test_image = tf.keras.preprocessing.image.img_to_array(test_image) / 255
	test_image = np.expand_dims(test_image, axis = 0)
	predicted_array = model.predict(test_image)
	predicted_value = output_class[np.argmax(predicted_array)]
	return predicted_value, data[predicted_value][0], data[predicted_value][1], data[predicted_value][2]

