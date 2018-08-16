#pragma once

class ErrorFormat
{
public:
    ErrorFormat() {}
    ~ErrorFormat() {}

    template <typename Type>
    ErrorFormat & operator << (const Type & value)
    {
        stream_ << value;
        return *this;
    }

    std::string str() const         { return stream_.str(); }
    operator std::string () const   { return stream_.str(); }

    enum ConvertToString 
    {
        to_str
    };
    std::string operator >> (ConvertToString) { return stream_.str(); }

private:
    std::stringstream stream_;

    ErrorFormat(const ErrorFormat &);
    ErrorFormat & operator = (ErrorFormat &);
};