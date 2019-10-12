#pragma once

#include <algorithm>
#include <vector>

template <typename TObserver>
class Observable {
 public:
  typedef size_t sizeType;

  typedef std::vector<TObserver*> ObserverList;

  /**
   * @brief Add an observer to the list.
   *
   * @param observer
   */
  void addObserver(TObserver *observer) {
    // See if we already have it in our list.
    typename ObserverList::const_iterator iObserver =
        std::find(observerList.begin(), observerList.end(), observer);

    // Not there?
    if (iObserver == observerList.end()) {
      // Add it.
      observerList.push_back(observer);
    }
  }

  /**
   * @brief Remove a particular observer from the list.
   *
   * @param observer
   */
  void removeObserver(TObserver *observer) {
    // See if we have it in our list.
    typename ObserverList::iterator iObserver =
        std::find(observerList.begin(), observerList.end(), observer);

    // Found it?
    if (iObserver != observerList.end()) {
      // Erase it.
      observerList.erase(iObserver);
    }
  }

  /**
   * @brief Clear all observers from the list.
   */
  void clearObservers() { observerList.clear(); }

  /**
   * @brief Returns the number of observers.
   *
   * @return sizeType
   */
  sizeType numberOfObservers() const { return observerList.size(); }

  /**
   * @brief Notify all of the observers, sending them the notification
   *
   * @tparam TNotification The notification type
   * @param n
   */
  template <typename TNotification>
  void notifyObservers(TNotification n) {
    for (size_t i = 0; i < observerList.size(); ++i) {
      observerList[i]->notify(n);
    }
  }

 private:
  //! The list of observers.
  ObserverList observerList;
};
