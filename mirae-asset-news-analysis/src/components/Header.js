import React from 'react';
import styled from 'styled-components';

const HeaderContainer = styled.header`
  background-color: white;
  color: #003366;
  padding: 20px;
  text-align: center;
  margin-bottom: 20px;
`;

const Logo = styled.img`
  max-width: 100%;
  height: auto;
`;

function Header() {
  return (
    <HeaderContainer>
      <Logo src="/ai-data-festival.png" alt="AI Data Festival" />
    </HeaderContainer>
  );
}

export default Header;