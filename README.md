# Recommend-events-in-social-network
Built a collaborative filtering recommender system to enhance the user experience of an event-based social network. The system is designed to recommend events that match a user's interest by analyzing their past behavior and preferences.

## Dataset

- User Demographic : user_id, locale, birthyear, gender, joinedAt, location, and timezone
- User Social Network: List of the user's friends' ids.
- Event details: event_id, user_id, start_time, city, state, zip, country, lat, and lng
- Event Description: Number of times the Nth most common word stem appears in the name or description of this event.

## Using Event Description (tf-idf rating)

Additionally, used K-means clustering to identify similar events based on the words in their description.  
Identified similarilites between events based on the tf-idf values of the keywords in the description, which helped to determine how relevant those words are to a given event.

## Dealing with coldstart problem
Implemented a hybrid approach to tackle the cold start problem that arises when new users join the platform or when there is not enough data to make personalized recommendations. 
This approach combines the results of top-n recommendations with user demographics to provide relevant suggestions to the user.
 
Reference: [Context-Aware Event Recommendation in Event-based Social Networks](https://homepages.dcc.ufmg.br/~rodrygo/wp-content/papercite-data/pdf/macedo2015recsys.pdf)  
Dataset: [Kaggle: Event Recommendation Engine Challenge](https://www.kaggle.com/competitions/event-recommendation-engine-challenge/data)


